from __future__ import annotations

"""
Prepare the WAF autoencoder training set from the raw request datasets in
data/training/ (benign requests only) and write:

  data/train/train.jsonl          - encoded benign requests for train_quick.py
  data/train/eval_benign.jsonl    - held-out benign request texts (calibration)
  data/train/eval_attack.jsonl    - attack request texts (calibration)
  models/checkpoints/vocab.json   - tokenizer vocabulary

The autoencoder is trained on benign traffic only; attack samples are never
used for training, only to calibrate the anomaly threshold afterwards.

Usage:
  PYTHONPATH=. python scripts/prepare_waf_data.py --maxlen 128 --vocab-size 5000
"""

import argparse
import glob
import json
import os
import random
from typing import Dict, List, Tuple

from src.preprocessing.compose import compose_request_text, parse_flat_request_line
from src.preprocessing.tokenizer import HTTPRequestTokenizer


def _entry_to_text(obj: Dict[str, object], default_label: str) -> Tuple[str, str]:
    """Return (label, composed_text) for one raw dataset entry."""
    label = str(obj.get("label") or default_label).lower()
    req = obj.get("request")
    if isinstance(req, dict):
        text = compose_request_text(
            str(req.get("method") or "GET"),
            str(req.get("path") or "/"),
            {str(k): str(v) for k, v in (req.get("query_params") or {}).items()},
            str(req.get("body") or ""),
        )
    else:
        flat = parse_flat_request_line(str(req or ""))
        text = compose_request_text(flat["method"], flat["path"], flat["query_params"], flat["body"])  # type: ignore[arg-type]
    return label, text


def load_raw(training_dir: str) -> Tuple[List[str], List[str]]:
    benign: List[str] = []
    attack: List[str] = []
    for fp in sorted(glob.glob(os.path.join(training_dir, "*.jsonl"))):
        # Files such as attack_large.jsonl carry no per-entry label; the file name is the label.
        default_label = "attack" if "attack" in os.path.basename(fp).lower() else "benign"
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                label, text = _entry_to_text(json.loads(line), default_label)
                (attack if label == "attack" else benign).append(text)
    # De-duplicate while preserving order
    benign = list(dict.fromkeys(benign))
    attack = list(dict.fromkeys(attack))
    return benign, attack


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--training-dir", default="data/training")
    ap.add_argument("--out", default="data/train/train.jsonl")
    ap.add_argument("--vocab", default="models/checkpoints/vocab.json")
    ap.add_argument("--vocab-size", type=int, default=5000)
    ap.add_argument("--maxlen", type=int, default=128)
    ap.add_argument("--holdout", type=float, default=0.15, help="Fraction of benign kept out for calibration")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    benign, attack = load_raw(args.training_dir)
    rng.shuffle(benign)
    n_hold = max(1, int(len(benign) * args.holdout))
    eval_benign, train_benign = benign[:n_hold], benign[n_hold:]

    tok = HTTPRequestTokenizer(vocab_size=args.vocab_size)
    tok.build_vocab(train_benign)
    os.makedirs(os.path.dirname(args.vocab) or ".", exist_ok=True)
    tok.save_vocab(args.vocab)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for text in train_benign:
            f.write(json.dumps(tok.encode(text, max_length=args.maxlen)) + "\n")
    out_dir = os.path.dirname(args.out) or "."
    with open(os.path.join(out_dir, "eval_benign.jsonl"), "w", encoding="utf-8") as f:
        for text in eval_benign:
            f.write(json.dumps({"text": text, "label": "benign"}) + "\n")
    with open(os.path.join(out_dir, "eval_attack.jsonl"), "w", encoding="utf-8") as f:
        for text in attack:
            f.write(json.dumps({"text": text, "label": "attack"}) + "\n")

    print(f"benign train: {len(train_benign)}  benign holdout: {len(eval_benign)}  attack (eval only): {len(attack)}")
    print(f"vocab tokens: {len(tok.token_to_id)} -> {args.vocab}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
