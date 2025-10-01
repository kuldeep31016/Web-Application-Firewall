from __future__ import annotations

"""
Create a delta benign dataset from new logs only (incremental).

Usage:
  PYTHONPATH=. python scripts/prepare_delta.py \
    --logs /path/to/new/logs \
    --out data/train/delta.jsonl \
    --vocab models/checkpoints/vocab.json \
    --maxlen 128
"""

import argparse
import json

from src.ingestion.batch_ingestion import batch_ingest_logs
from src.preprocessing.parser import parse_request
from src.preprocessing.normalizer import normalize_path, normalize_params, replace_dynamic_values
from src.preprocessing.tokenizer import HTTPRequestTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True)
    ap.add_argument("--out", default="data/train/delta.jsonl")
    ap.add_argument("--vocab", default="models/checkpoints/vocab.json")
    ap.add_argument("--maxlen", type=int, default=128)
    args = ap.parse_args()

    tmp = "data/processed/_tmp_delta.jsonl"
    batch_ingest_logs(args.logs, tmp)

    tok = HTTPRequestTokenizer()
    tok.load_vocab(args.vocab)

    with open(tmp, "r", encoding="utf-8") as f, open(args.out, "w", encoding="utf-8") as out:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            req = parse_request(entry)
            npath = normalize_path(req.path)
            nparams = normalize_params(req.query_params)
            composed = f"{req.method} {npath}"
            if nparams:
                query = "&".join(f"{k}={v}" for k, v in nparams.items())
                composed += f"?{query}"
            if req.body:
                composed += " BODY:" + replace_dynamic_values(req.body)
            enc = tok.encode(composed, max_length=args.maxlen)
            out.write(json.dumps(enc) + "\n")
    print("Wrote delta:", args.out)


if __name__ == "__main__":
    main()


