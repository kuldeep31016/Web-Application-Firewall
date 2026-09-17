from __future__ import annotations

"""
Run the missing-information sweep and the feature-dependency ablation from the
command line, store them (same SQLite store as the API) and print Markdown
tables. Reproducible for a given --seed, --sample-size and model checkpoints.

Usage:
  PYTHONPATH=. python scripts/run_robustness_experiments.py
  PYTHONPATH=. python scripts/run_robustness_experiments.py --sample-size 20000 --seed 7 --out results.json
  PYTHONPATH=. python scripts/run_robustness_experiments.py --features scheme subdomain   # always-missing segments
"""

import argparse
import json
import sys
from typing import Dict, List

from src.phishing.experiments import DEFAULT_RATES
from src.phishing.service import PhishingService


def _pct(x) -> str:
    return "—" if x is None else f"{x * 100:.2f}%"


def _signed(x) -> str:
    return "—" if x is None else f"{x * 100:+.2f} pp"


def sweep_table(exp: Dict[str, object]) -> str:
    lines = [
        "| Condition | Strategy | Missing | Realised | Accuracy | Precision | Recall | F1 | FPR | FNR | ROC AUC | Δ acc |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in exp["results"]:  # type: ignore[index]
        d = r["delta_vs_complete"]
        lines.append(
            f"| {r['condition']} | {r['strategy']} | {_pct(r['missing_rate'])} | {_pct(r['realised_missing_rate'])} | {_pct(r['accuracy'])} | "
            f"{_pct(r['precision'])} | {_pct(r['recall'])} | {_pct(r['f1'])} | {_pct(r['fpr'])} | {_pct(r['fnr'])} | "
            f"{'—' if r['roc_auc'] is None else f'{r['roc_auc']:.4f}'} | {_signed(d['accuracy'])} |"
        )
    return "\n".join(lines)


def impact_table(exp: Dict[str, object]) -> str:
    presence = exp["summary"]["presence_rate"]  # type: ignore[index]
    lines = ["| Removed | Present in test URLs | Strategy | Accuracy | F1 | Δ acc | Δ F1 |", "|---|---:|---|---:|---:|---:|---:|"]
    for r in exp["results"]:  # type: ignore[index]
        if r["condition"] == "complete":
            continue
        feats = r["features"]
        pres = _pct(presence[feats[0]]) if len(feats) == 1 else "—"
        d = r["delta_vs_complete"]
        lines.append(f"| {' + '.join(feats)} | {pres} | {r['strategy']} | {_pct(r['accuracy'])} | {_pct(r['f1'])} | {_signed(d['accuracy'])} | {_signed(d['f1'])} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rates", type=float, nargs="+", default=list(DEFAULT_RATES))
    ap.add_argument("--features", nargs="*", default=[], help="Segments that are always missing in the sweep")
    ap.add_argument("--strategies", nargs="*", default=None)
    ap.add_argument("--sample-size", type=int, default=5000, help="Stratified test sample; 0 = whole test split")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-impact", action="store_true")
    ap.add_argument("--no-store", action="store_true", help="Do not persist to logs/detections.db")
    ap.add_argument("--out", default=None, help="Write both experiments as JSON")
    args = ap.parse_args()

    svc = PhishingService()
    loaded = svc.load_models()
    if not any(loaded.values()):
        sys.exit("No URL classifier found. Train first: PYTHONPATH=. python scripts/train_url_classifier.py")
    sample = None if args.sample_size == 0 else args.sample_size

    sweep = svc.run_missing_information(
        persist=not args.no_store, rates=args.rates, features=args.features, strategies=args.strategies, sample_size=sample, seed=args.seed
    )
    print(f"\n## Missing-information sweep  (experiment {sweep['experiment_id']}, {sweep['sample_size']} test URLs, seed {args.seed})\n")
    print(sweep_table(sweep))

    out: Dict[str, object] = {"missing_information": sweep}
    if not args.skip_impact:
        impact = svc.run_feature_impact(persist=not args.no_store, strategies=args.strategies, sample_size=sample, seed=args.seed)
        print(f"\n## Feature dependency  (experiment {impact['experiment_id']})\n")
        print(impact_table(impact))
        out["feature_impact"] = impact

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
