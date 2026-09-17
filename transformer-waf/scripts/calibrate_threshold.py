from __future__ import annotations

"""
Calibrate the WAF anomaly threshold on held-out data produced by
scripts/prepare_waf_data.py.

Scores held-out benign requests and attack requests with the trained
autoencoder, prints the score distributions, and reports the detection
quality (TPR / FPR) for a range of candidate thresholds. With --write the
chosen threshold is written to config.yaml under detection.threshold.

Usage:
  PYTHONPATH=. python scripts/calibrate_threshold.py            # report only
  PYTHONPATH=. python scripts/calibrate_threshold.py --write    # also update config.yaml
"""

import argparse
import json
import os
import re
from typing import List

from src.models.inference import InferenceEngine
from src.preprocessing.tokenizer import HTTPRequestTokenizer
from src.utils.config import CONFIG_PATH, load_config


def _load_texts(path: str) -> List[str]:
    out: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line)["text"])
    return out


def _score(engine: InferenceEngine, tok: HTTPRequestTokenizer, texts: List[str], max_len: int, batch: int = 64) -> List[float]:
    scores: List[float] = []
    for i in range(0, len(texts), batch):
        chunk = texts[i : i + batch]
        enc = [tok.encode(t, max_length=max_len) for t in chunk]
        res = engine.predict_batch([e["input_ids"] for e in enc], [e["attention_mask"] for e in enc])
        scores.extend(float(r["anomaly_score"]) for r in res)
    return scores


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    idx = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[idx]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benign", default="data/train/eval_benign.jsonl")
    ap.add_argument("--attack", default="data/train/eval_attack.jsonl")
    ap.add_argument("--model", default="models/checkpoints/best.pt")
    ap.add_argument("--vocab", default="models/checkpoints/vocab.json")
    ap.add_argument("--maxlen", type=int, default=128)
    ap.add_argument("--benign-percentile", type=float, default=0.95, help="Benign score percentile used as threshold")
    ap.add_argument("--write", action="store_true", help="Write the chosen threshold into config.yaml")
    args = ap.parse_args()

    cfg = load_config(CONFIG_PATH)
    engine = InferenceEngine(model_path=args.model, threshold=float(cfg.get("detection", {}).get("threshold", 0.75)))
    engine.load_model()
    tok = HTTPRequestTokenizer(vocab_size=int(cfg.get("model", {}).get("vocab_size", 10000)))
    tok.load_vocab(args.vocab)

    benign = _load_texts(args.benign)
    attack = _load_texts(args.attack)
    b = _score(engine, tok, benign, args.maxlen)
    a = _score(engine, tok, attack, args.maxlen)

    print(f"benign  n={len(b)}  p50={_percentile(b,0.5):.3f}  p95={_percentile(b,0.95):.3f}  p99={_percentile(b,0.99):.3f}  max={max(b):.3f}")
    print(f"attack  n={len(a)}  p05={_percentile(a,0.05):.3f}  p50={_percentile(a,0.5):.3f}  min={min(a):.3f}")
    print()
    print(f"{'threshold':>10} {'TPR(attack)':>12} {'FPR(benign)':>12}")
    chosen = _percentile(b, args.benign_percentile)
    for q in (0.90, 0.95, 0.99):
        thr = _percentile(b, q)
        tpr = sum(1 for s in a if s > thr) / max(len(a), 1)
        fpr = sum(1 for s in b if s > thr) / max(len(b), 1)
        marker = "  <- chosen" if abs(thr - chosen) < 1e-9 else ""
        print(f"{thr:>10.3f} {tpr:>12.3f} {fpr:>12.3f}{marker}")

    if args.write:
        value = round(float(chosen), 4)
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            text = f.read()
        # Replace only the threshold line so the rest of config.yaml keeps its formatting
        new_text, count = re.subn(r"^(\s*threshold:\s*)[0-9.eE+-]+", rf"\g<1>{value}", text, count=1, flags=re.MULTILINE)
        if count != 1:
            raise SystemExit("could not find 'threshold:' in config.yaml")
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            f.write(new_text)
        print(f"\nwrote detection.threshold={value} to {os.path.abspath(CONFIG_PATH)}")


if __name__ == "__main__":
    main()
