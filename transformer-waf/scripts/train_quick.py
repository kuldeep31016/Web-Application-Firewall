from __future__ import annotations

"""
Quick training script:
- Loads JSONL dataset from data/train/train.jsonl (created by scripts/generate_benign.py)
- Trains a small Transformer autoencoder for a few epochs
- Saves checkpoint to models/checkpoints/best.pt

Usage:
  PYTHONPATH=. python scripts/train_quick.py --epochs 3 --batch 64
"""

import argparse
import json
import os
import random
from typing import List

import torch
from torch.utils.data import DataLoader

from src.models.transformer_model import WAFTransformer
from src.models.train import SequenceDataset, train_model


def load_jsonl(path: str) -> SequenceDataset:
    sequences: List[List[int]] = []
    masks: List[List[int]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            sequences.append(obj["input_ids"])  # type: ignore[index]
            masks.append(obj["attention_mask"])  # type: ignore[index]
    return SequenceDataset(sequences, masks)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/train/train.jsonl")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--vocab", type=int, default=5000)
    parser.add_argument("--embed", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--ff", type=int, default=256)
    parser.add_argument("--maxlen", type=int, default=128)
    args = parser.parse_args()

    ds = load_jsonl(args.data)
    n = len(ds)
    idx = list(range(n))
    random.shuffle(idx)
    split = max(1, int(0.8 * n))
    train_idx, val_idx = idx[:split], idx[split:]

    train_ds = SequenceDataset([ds.sequences[i] for i in train_idx], [ds.attention_masks[i] for i in train_idx])
    val_ds = SequenceDataset([ds.sequences[i] for i in val_idx], [ds.attention_masks[i] for i in val_idx])

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch)

    model = WAFTransformer(
        vocab_size=args.vocab,
        embed_dim=args.embed,
        num_heads=args.heads,
        num_layers=args.layers,
        ff_dim=args.ff,
        dropout=0.1,
        max_len=args.maxlen,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_model(model, train_loader, val_loader, epochs=args.epochs, device=device)

    os.makedirs("models/checkpoints", exist_ok=True)
    ckpt = {
        "model": model.state_dict(),
        "meta": {
            "vocab_size": args.vocab,
            "embed_dim": args.embed,
            "num_heads": args.heads,
            "num_layers": args.layers,
            "ff_dim": args.ff,
            "dropout": 0.1,
            "max_len": args.maxlen,
        },
    }
    torch.save(ckpt, "models/checkpoints/best.pt")
    print("Saved models/checkpoints/best.pt")


if __name__ == "__main__":
    main()



