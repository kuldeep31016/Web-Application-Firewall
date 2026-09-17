"""
Application-level service that ties the URL classifier, the missingness
simulation, the experiments and the persistence together. The API layer is a
thin wrapper over this class so that scripts and tests can use it directly.
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Callable, Dict, List, Optional

import pandas as pd
import torch

from ..models.url_classifier import URLClassifierEngine, load_url_classifier
from ..preprocessing.tokenizer import HTTPRequestTokenizer
from ..storage.experiment_store import ExperimentStore, experiment_store
from ..utils.logger import logger
from .dataset import DEFAULT_DATASET_PATH, DatasetError, DatasetInfo, _file_sha256, dataset_info, load_url_dataset
from .experiments import STRATEGIES, ExperimentError, ExperimentRunner
from .missingness import MissingnessConfig, apply_missingness
from .training import MODELS_DIR, VARIANTS, model_path_for, vocab_path_for
from .url_features import (
    FEATURE_DESCRIPTIONS,
    FEATURE_NAMES,
    describe_segments,
    normalize_feature_names,
    split_url,
    validate_url,
)


# Optional hook the detection API installs so a URL analysis can also report
# the WAF anomaly score of the equivalent request. Signature:
#   (method, path, query_params, body) -> {"anomaly_score": float, "is_anomaly": bool, "threshold": float} | None
WafScorer = Callable[[str, str, Dict[str, str], str], Optional[Dict[str, object]]]


class PhishingService:
    def __init__(
        self,
        dataset_path: str = DEFAULT_DATASET_PATH,
        threshold: float = 0.5,
        store: Optional[ExperimentStore] = None,
        models_dir: str = MODELS_DIR,
    ) -> None:
        self.dataset_path = dataset_path
        self.models_dir = models_dir
        self.threshold = float(threshold)
        self.store = store or experiment_store
        self.engines: Dict[str, URLClassifierEngine] = {}
        self._df: Optional[pd.DataFrame] = None
        self._dataset_info: Optional[DatasetInfo] = None
        self._waf_scorer: Optional[WafScorer] = None
        self.device = torch.device("cpu")

    # ----- lifecycle ----------------------------------------------------------

    def set_waf_scorer(self, scorer: Optional[WafScorer]) -> None:
        self._waf_scorer = scorer

    def load_models(self) -> Dict[str, bool]:
        loaded: Dict[str, bool] = {}
        vocab_path = vocab_path_for(self.models_dir)
        if not os.path.exists(vocab_path):
            logger.warning("URL classifier vocabulary not found at {}", vocab_path)
            return {v: False for v in VARIANTS}
        tokenizer = HTTPRequestTokenizer()
        tokenizer.load_vocab(vocab_path)
        vocab_hash = _file_sha256(vocab_path)
        self.vocab_mismatch: Dict[str, bool] = {}
        for variant in VARIANTS:
            path = model_path_for(variant, self.models_dir)
            if not os.path.exists(path):
                loaded[variant] = False
                continue
            try:
                model = load_url_classifier(path, self.device)
                expected = model.meta.get("vocab_sha256")
                self.vocab_mismatch[variant] = bool(expected and expected != vocab_hash)
                if self.vocab_mismatch[variant]:
                    logger.warning("URL classifier {} was trained with a different vocabulary than {}; retrain it.", variant, vocab_path)
                self.engines[variant] = URLClassifierEngine(model, tokenizer, threshold=self.threshold, device=self.device)
                loaded[variant] = True
            except Exception as exc:  # corrupt checkpoint: report, do not crash the API
                logger.exception("Failed to load URL classifier {}: {}", variant, exc)
                loaded[variant] = False
        return loaded

    def reload(self) -> Dict[str, bool]:
        self.engines = {}
        self._df = None
        self._dataset_info = None
        return self.load_models()

    @property
    def ready(self) -> bool:
        return bool(self.engines)

    def dataframe(self) -> pd.DataFrame:
        if self._df is None:
            self._df = load_url_dataset(self.dataset_path)
            self._dataset_info = dataset_info(self.dataset_path, self._df)
        return self._df

    def dataset(self) -> DatasetInfo:
        self.dataframe()
        assert self._dataset_info is not None
        return self._dataset_info

    def dataset_available(self) -> bool:
        return os.path.exists(self.dataset_path)

    # ----- status ------------------------------------------------------------

    def status(self) -> Dict[str, object]:
        models: Dict[str, object] = {}
        for variant in VARIANTS:
            engine = self.engines.get(variant)
            meta = engine.model.meta if engine else {}
            models[variant] = {
                "loaded": engine is not None,
                "path": os.path.relpath(model_path_for(variant, self.models_dir), os.getcwd()) if os.path.exists(model_path_for(variant, self.models_dir)) else None,
                "trained_at": meta.get("trained_at"),
                "n_train": meta.get("n_train"),
                "best_val_f1": meta.get("best_val_f1"),
                "mitigation": meta.get("mitigation"),
                "train_config": meta.get("train_config"),
                "dataset_sha256": meta.get("dataset_sha256"),
                "vocab_mismatch": getattr(self, "vocab_mismatch", {}).get(variant, False),
            }
        ds: Optional[Dict[str, object]] = None
        if self.dataset_available():
            try:
                ds = self.dataset().to_dict()
            except DatasetError as exc:
                ds = {"error": str(exc)}
        return {
            "ready": self.ready,
            "threshold": self.threshold,
            "features": [{"name": f, "description": FEATURE_DESCRIPTIONS[f]} for f in FEATURE_NAMES],
            "strategies": [
                {"name": s.name, "model_variant": s.model_variant, "representation": s.representation, "description": s.description, "available": s.model_variant in self.engines}
                for s in STRATEGIES.values()
            ],
            "models": models,
            "dataset": ds,
            "waf_available": self._waf_scorer is not None,
        }

    # ----- single URL --------------------------------------------------------

    def analyze_url(self, url: str, missing_features: Optional[List[str]] = None, strategy: str = "none", persist: bool = True) -> Dict[str, object]:
        text = validate_url(url)
        missing = normalize_feature_names(missing_features)
        if strategy not in STRATEGIES:
            raise ValueError(f"Unknown strategy {strategy!r}; valid: {list(STRATEGIES)}")
        strat = STRATEGIES[strategy]
        engine = self.engines.get(strat.model_variant)
        if engine is None:
            raise ExperimentError(f"The {strat.model_variant!r} URL classifier is not trained/loaded")

        segments = split_url(text)
        config = MissingnessConfig(features=tuple(sorted(missing)), rate=0.0, representation=strat.representation)
        masked = apply_missingness([segments], config)[0]
        started = time.perf_counter()
        prob = engine.score([masked.text])[0]
        inference_ms = (time.perf_counter() - started) * 1000.0

        waf: Optional[Dict[str, object]] = None
        if self._waf_scorer is not None:
            query = dict(p.split("=", 1) if "=" in p else (p, "") for p in segments.query.split("&") if p)
            try:
                waf = self._waf_scorer("GET", segments.path or "/", query, "")
            except Exception as exc:  # WAF failure must not break URL analysis
                logger.warning("WAF scoring failed for URL analysis: {}", exc)
                waf = None

        record = {
            "request_id": str(uuid.uuid4()),
            "timestamp": time.time(),
            "url": text,
            "model_text": masked.text,
            "segments": segments.as_dict(),
            "present_features": segments.present_features(),
            "missing_features": sorted(missing),
            "unavailable_present_features": sorted(masked.removed_present),
            "strategy": strat.name,
            "representation": strat.representation,
            "model_variant": strat.model_variant,
            "model_trained_at": engine.model.meta.get("trained_at"),
            "phishing_probability": float(prob),
            "is_phishing": bool(prob >= engine.threshold),
            "threshold": engine.threshold,
            "inference_ms": round(inference_ms, 3),
            "descriptive": describe_segments(segments),
            "waf": waf,
        }
        if persist:
            self.store.store_url_detection(
                {
                    **record,
                    "waf_anomaly_score": waf.get("anomaly_score") if waf else None,
                    "waf_is_anomaly": waf.get("is_anomaly") if waf else None,
                }
            )
        return record

    # ----- experiments -------------------------------------------------------

    def runner(self) -> ExperimentRunner:
        if not self.engines:
            raise ExperimentError("No URL classifier is loaded. Train with scripts/train_url_classifier.py.")
        return ExperimentRunner(self.engines, self.dataframe(), self.dataset())

    def run_missing_information(self, persist: bool = True, **kwargs) -> Dict[str, object]:
        experiment = self.runner().run_missing_information(**kwargs)
        if persist:
            self.store.store_experiment(experiment, experiment["results"])
        return experiment

    def run_feature_impact(self, persist: bool = True, **kwargs) -> Dict[str, object]:
        experiment = self.runner().run_feature_impact(**kwargs)
        if persist:
            self.store.store_experiment(experiment, experiment["results"])
        return experiment


__all__ = ["PhishingService", "WafScorer"]
