from __future__ import annotations

"""
Train the Transformer URL classifier(s) used by the robustness experiments.

  baseline - trained on complete URLs
  robust   - trained with missing-information augmentation (the mitigation)

Both variants share models/phishing/vocab.json and the dataset split in
data/phishing/urls.csv (create it with scripts/fetch_phishing_dataset.py).

Usage:
  PYTHONPATH=. python scripts/train_url_classifier.py --variant both --epochs 3
  PYTHONPATH=. python scripts/train_url_classifier.py --variant baseline --max-train-samples 20000 --epochs 2
"""

import argparse
import json

from src.phishing.training import TrainConfig, VARIANTS, train_url_classifier


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(VARIANTS) + ["both"], default="both")
    ap.add_argument("--dataset", default=None, help="Path to url/label/split CSV (default: data/phishing/urls.csv)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--maxlen", type=int, default=48)
    ap.add_argument("--vocab-size", type=int, default=8000)
    ap.add_argument("--embed", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--ff", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-train-samples", type=int, default=None)
    ap.add_argument("--aug-max-rate", type=float, default=0.5, help="Upper bound of the per-sample missing rate used for augmentation")
    ap.add_argument("--rebuild-vocab", action="store_true")
    args = ap.parse_args()

    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        max_len=args.maxlen,
        vocab_size=args.vocab_size,
        embed_dim=args.embed,
        num_heads=args.heads,
        num_layers=args.layers,
        ff_dim=args.ff,
        seed=args.seed,
        max_train_samples=args.max_train_samples,
        aug_max_rate=args.aug_max_rate,
    )
    variants = list(VARIANTS) if args.variant == "both" else [args.variant]
    for i, variant in enumerate(variants):
        summary = train_url_classifier(
            variant, config, dataset_path=args.dataset, rebuild_vocab=(args.rebuild_vocab and i == 0)
        )
        print(json.dumps({k: v for k, v in summary.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
