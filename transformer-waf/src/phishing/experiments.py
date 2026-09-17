"""
Robustness experiments: how does detection performance change when URL
information is unavailable, and does the mitigation recover it?

Handling strategies (what the system does when information is missing):

    none                 the segment is simply blank; the baseline model
                         (trained on complete URLs) makes the decision
    missing_token        the segment is replaced by an explicit ``[MISSING]``
                         marker; still the baseline model
    augmented_training   ``[MISSING]`` marker + the robust model, which was
                         trained with missing-information augmentation
                         (this is the mitigation under test)

Every condition is evaluated on the *same* held-out test URLs with the *same*
seeded missingness draw, so differences between strategies are attributable
to the strategy alone. Nothing is measured on training data.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd

from ..models.url_classifier import URLClassifierEngine
from .dataset import DatasetInfo, sample_split
from .metrics import compute_metrics, metric_delta
from .missingness import MissingnessConfig, apply_missingness, realised_missing_rate
from .url_features import FEATURE_NAMES, URLSegments, split_url


@dataclass(frozen=True)
class Strategy:
    name: str
    model_variant: str
    representation: str
    description: str


STRATEGIES: Dict[str, Strategy] = {
    "none": Strategy("none", "baseline", "blank", "Missing segments are blanked; baseline model decides."),
    "missing_token": Strategy(
        "missing_token", "baseline", "missing_token", "Missing segments become an explicit [MISSING] marker; baseline model decides."
    ),
    "augmented_training": Strategy(
        "augmented_training",
        "robust",
        "missing_token",
        "[MISSING] marker + model trained with missing-information augmentation (mitigation).",
    ),
}
DEFAULT_RATES = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
MAX_SAMPLE_SIZE = 50000


class ExperimentError(RuntimeError):
    pass


class ExperimentRunner:
    def __init__(self, engines: Dict[str, URLClassifierEngine], df: pd.DataFrame, dataset: DatasetInfo) -> None:
        self.engines = engines
        self.df = df
        self.dataset = dataset

    # ----- helpers ----------------------------------------------------------

    def _strategies(self, names: Optional[Iterable[str]]) -> List[Strategy]:
        chosen = list(names) if names else list(STRATEGIES)
        out: List[Strategy] = []
        for name in chosen:
            if name not in STRATEGIES:
                raise ExperimentError(f"Unknown strategy {name!r}; valid: {list(STRATEGIES)}")
            strat = STRATEGIES[name]
            if strat.model_variant not in self.engines:
                raise ExperimentError(
                    f"Strategy {name!r} needs the {strat.model_variant!r} model, which is not trained. "
                    "Run scripts/train_url_classifier.py first."
                )
            out.append(strat)
        return out

    def _sample(self, sample_size: Optional[int], seed: int, split: str = "test") -> pd.DataFrame:
        if sample_size is not None and (sample_size < 10 or sample_size > MAX_SAMPLE_SIZE):
            raise ExperimentError(f"sample_size must be between 10 and {MAX_SAMPLE_SIZE}")
        part = sample_split(self.df, split, sample_size, seed)
        if part.empty:
            raise ExperimentError(f"No samples in split {split!r}")
        return part

    def evaluate_condition(
        self, segments: Sequence[URLSegments], labels: Sequence[int], features: Sequence[str], rate: float, strategy: Strategy, seed: int
    ) -> Dict[str, object]:
        config = MissingnessConfig(features=tuple(sorted(features)), rate=rate, representation=strategy.representation, seed=seed)
        masked = apply_missingness(segments, config)
        engine = self.engines[strategy.model_variant]
        scores = engine.score([m.text for m in masked])
        preds = [1 if s >= engine.threshold else 0 for s in scores]
        metrics = compute_metrics(labels, preds, scores)
        metrics.update(
            {
                "strategy": strategy.name,
                "model_variant": strategy.model_variant,
                "representation": strategy.representation,
                "missing_rate": rate,
                "features": list(config.features),
                "realised_missing_rate": realised_missing_rate(masked),
                "threshold": engine.threshold,
            }
        )
        return metrics

    # ----- experiments ------------------------------------------------------

    def run_missing_information(
        self,
        rates: Sequence[float] = DEFAULT_RATES,
        features: Sequence[str] = (),
        strategies: Optional[Iterable[str]] = None,
        sample_size: Optional[int] = 5000,
        seed: int = 42,
        split: str = "test",
    ) -> Dict[str, object]:
        """Sweep missing-information levels for each handling strategy."""
        rates = sorted({float(r) for r in rates})
        if not rates:
            raise ExperimentError("At least one missing rate is required")
        if any(r < 0 or r > 1 for r in rates):
            raise ExperimentError("Missing rates must be between 0 and 1")
        strats = self._strategies(strategies)
        started = time.time()
        part = self._sample(sample_size, seed, split)
        segments = [split_url(u) for u in part["url"].tolist()]
        labels = part["label"].tolist()

        # Reference: complete information, baseline model
        reference_strategy = STRATEGIES["none"] if "baseline" in self.engines else strats[0]
        reference = self.evaluate_condition(segments, labels, (), 0.0, reference_strategy, seed)
        reference["condition"] = "complete"
        reference["delta_vs_complete"] = metric_delta(reference, reference)
        results: List[Dict[str, object]] = [reference]

        for rate in rates:
            for strat in strats:
                if rate == 0.0 and not features and strat is reference_strategy:
                    continue  # identical to the reference row
                res = self.evaluate_condition(segments, labels, features, rate, strat, seed)
                if rate == 0.0 and not features:
                    res["condition"] = "complete"
                elif strat.name == "augmented_training":
                    res["condition"] = "mitigated"
                else:
                    res["condition"] = "incomplete"
                res["delta_vs_complete"] = metric_delta(reference, res)
                results.append(res)

        # Recovery: mitigated minus the 'none' strategy at the same rate
        by_key = {(r["strategy"], r["missing_rate"]): r for r in results if r.get("condition") != "complete" or r["missing_rate"] == 0.0}
        for r in results:
            if r["strategy"] == "augmented_training":
                base = by_key.get(("none", r["missing_rate"]))
                r["recovery_vs_none"] = metric_delta(base, r) if base else None

        summary = self._summarise_sweep(results)
        experiment = {
            "experiment_id": str(uuid.uuid4()),
            "kind": "missing_information",
            "created_at": time.time(),
            "seed": seed,
            "dataset_path": self.dataset.path,
            "dataset_sha256": self.dataset.sha256,
            "evaluation_split": split,
            "sample_size": int(len(part)),
            "config": {"rates": rates, "features": list(features), "strategies": [s.name for s in strats]},
            "models": self._model_provenance(),
            "summary": summary,
            "duration_seconds": round(time.time() - started, 2),
            "results": results,
        }
        return experiment

    def run_feature_impact(
        self,
        features: Optional[Sequence[str]] = None,
        combinations: Optional[Sequence[Sequence[str]]] = None,
        strategies: Optional[Iterable[str]] = None,
        sample_size: Optional[int] = 5000,
        seed: int = 42,
        split: str = "test",
    ) -> Dict[str, object]:
        """Remove each information unit (and optional combinations) in turn."""
        singles = [[f] for f in (features or FEATURE_NAMES)]
        combos = [list(c) for c in (combinations or []) if c]
        strats = self._strategies(strategies)
        started = time.time()
        part = self._sample(sample_size, seed, split)
        segments = [split_url(u) for u in part["url"].tolist()]
        labels = part["label"].tolist()
        presence = {f: sum(1 for s in segments if s.get(f)) / len(segments) for f in FEATURE_NAMES}

        reference_strategy = STRATEGIES["none"] if "baseline" in self.engines else strats[0]
        reference = self.evaluate_condition(segments, labels, (), 0.0, reference_strategy, seed)
        reference["condition"] = "complete"
        reference["delta_vs_complete"] = metric_delta(reference, reference)
        results: List[Dict[str, object]] = [reference]
        for feats in singles + combos:
            for strat in strats:
                res = self.evaluate_condition(segments, labels, feats, 0.0, strat, seed)
                res["condition"] = "mitigated" if strat.name == "augmented_training" else "incomplete"
                res["delta_vs_complete"] = metric_delta(reference, res)
                res["extra"] = {"presence_rate": {f: presence[f] for f in feats}}
                results.append(res)

        ranking = sorted(
            (
                {"features": r["features"], "accuracy_drop": -r["delta_vs_complete"]["accuracy"], "f1_drop": -r["delta_vs_complete"]["f1"]}
                for r in results
                if r["strategy"] == "none" and r["condition"] != "complete"
            ),
            key=lambda x: x["f1_drop"],
            reverse=True,
        )
        experiment = {
            "experiment_id": str(uuid.uuid4()),
            "kind": "feature_impact",
            "created_at": time.time(),
            "seed": seed,
            "dataset_path": self.dataset.path,
            "dataset_sha256": self.dataset.sha256,
            "evaluation_split": split,
            "sample_size": int(len(part)),
            "config": {"features": [f[0] for f in singles], "combinations": combos, "strategies": [s.name for s in strats]},
            "models": self._model_provenance(),
            "summary": {"presence_rate": presence, "ranking_by_f1_drop": ranking, "reference": _scalar(reference)},
            "duration_seconds": round(time.time() - started, 2),
            "results": results,
        }
        return experiment

    # ----- provenance / summary --------------------------------------------

    def _model_provenance(self) -> Dict[str, object]:
        out: Dict[str, object] = {}
        for variant, engine in self.engines.items():
            meta = engine.model.meta
            out[variant] = {
                "trained_at": meta.get("trained_at"),
                "n_train": meta.get("n_train"),
                "dataset_sha256": meta.get("dataset_sha256"),
                "train_config": meta.get("train_config"),
                "best_val_f1": meta.get("best_val_f1"),
                "threshold": engine.threshold,
            }
        return out

    @staticmethod
    def _summarise_sweep(results: List[Dict[str, object]]) -> Dict[str, object]:
        reference = next(r for r in results if r["condition"] == "complete" and r["strategy"] == "none") if any(
            r["strategy"] == "none" for r in results
        ) else results[0]
        worst = {}
        for r in results:
            if r["condition"] == "complete":
                continue
            key = r["strategy"]
            if key not in worst or r["accuracy"] < worst[key]["accuracy"]:
                worst[key] = _scalar(r)
        return {"reference": _scalar(reference), "worst_by_strategy": worst}


def _scalar(r: Dict[str, object]) -> Dict[str, object]:
    return {k: r.get(k) for k in ("strategy", "missing_rate", "features", "accuracy", "precision", "recall", "f1", "fpr", "fnr", "roc_auc")}


__all__ = ["STRATEGIES", "DEFAULT_RATES", "MAX_SAMPLE_SIZE", "Strategy", "ExperimentError", "ExperimentRunner"]
