"""
Training of the Transformer URL classifier, with and without the
missing-information mitigation.

Two model variants share one vocabulary and one train/val/test split:

* ``baseline`` - trained on complete URLs only.
* ``robust``   - trained with *missing-information augmentation*: in every
                 epoch each training URL has a fresh, random subset of its
                 information units replaced by ``[MISSING]`` (per-sample rate
                 drawn uniformly from [0, aug_max_rate]). The model therefore
                 learns to make a decision from whatever information remains.

The mitigation never touches the test split; evaluation of both variants uses
exactly the same masked test samples.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ..models.url_classifier import TransformerURLClassifier, save_url_classifier
from ..preprocessing.tokenizer import HTTPRequestTokenizer
from ..utils.logger import logger
from .dataset import DEFAULT_DATASET_PATH, PACKAGE_ROOT, _file_sha256, dataset_info, load_url_dataset
from .metrics import compute_metrics
from .url_features import FEATURE_NAMES, URLSegments, compose_url_text, split_url


MODELS_DIR = os.path.join(PACKAGE_ROOT, "models", "phishing")
VOCAB_PATH = os.path.join(MODELS_DIR, "vocab.json")
VARIANTS = ("baseline", "robust")


def model_path_for(variant: str, models_dir: str = MODELS_DIR) -> str:
    return os.path.join(models_dir, f"url_classifier_{variant}.pt")


def vocab_path_for(models_dir: str = MODELS_DIR) -> str:
    return os.path.join(models_dir, "vocab.json")


@dataclass
class TrainConfig:
    epochs: int = 3
    batch_size: int = 256
    lr: float = 5e-4
    max_len: int = 48
    vocab_size: int = 8000
    embed_dim: int = 128
    num_heads: int = 4
    num_layers: int = 3
    ff_dim: int = 256
    dropout: float = 0.1
    seed: int = 42
    max_train_samples: Optional[int] = None
    aug_max_rate: float = 0.5
    aug_representation: str = "missing_token"
    device: str = "cpu"

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _segments(urls: List[str]) -> List[URLSegments]:
    return [split_url(u) for u in urls]


def build_or_load_vocab(train_urls: List[str], vocab_size: int, path: str = VOCAB_PATH, rebuild: bool = False) -> HTTPRequestTokenizer:
    tok = HTTPRequestTokenizer(vocab_size=vocab_size)
    if os.path.exists(path) and not rebuild:
        tok.load_vocab(path)
        return tok
    texts = [compose_url_text(s) for s in _segments(train_urls)]
    tok.build_vocab(texts)
    tok.save_vocab(path)
    return tok


def augment_texts(segments: List[URLSegments], max_rate: float, representation: str, rng: np.random.RandomState) -> List[str]:
    """Missing-information augmentation: per-sample rate ~ U(0, max_rate), each
    unit removed independently with that rate."""
    rates = rng.uniform(0.0, max_rate, size=(len(segments), 1))
    hits = rng.random_sample((len(segments), len(FEATURE_NAMES))) < rates
    out: List[str] = []
    for seg, row in zip(segments, hits):
        missing = {name for name, hit in zip(FEATURE_NAMES, row) if hit}
        out.append(compose_url_text(seg, missing, representation=representation))
    return out


def _encode_all(tok: HTTPRequestTokenizer, texts: List[str], max_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    enc = [tok.encode(t, max_length=max_len) for t in texts]
    ids = torch.tensor([e["input_ids"] for e in enc], dtype=torch.long)
    mask = torch.tensor([e["attention_mask"] for e in enc], dtype=torch.long)
    return ids, mask


@torch.no_grad()
def _evaluate(model: TransformerURLClassifier, ids: torch.Tensor, mask: torch.Tensor, labels: np.ndarray, device: torch.device, batch: int = 512) -> Dict[str, object]:
    model.eval()
    probs: List[float] = []
    for i in range(0, len(ids), batch):
        p = model.predict_proba(ids[i : i + batch].to(device), mask[i : i + batch].to(device))
        probs.extend(p.cpu().tolist())
    preds = [1 if p >= 0.5 else 0 for p in probs]
    return compute_metrics(labels.tolist(), preds, probs)


def train_url_classifier(
    variant: str,
    config: TrainConfig,
    df: Optional[pd.DataFrame] = None,
    dataset_path: Optional[str] = None,
    out_path: Optional[str] = None,
    rebuild_vocab: bool = False,
    models_dir: str = MODELS_DIR,
) -> Dict[str, object]:
    """Train one variant and save its checkpoint. Returns training summary."""
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    _set_seed(config.seed)
    device = torch.device(config.device)
    dataset_path = dataset_path or DEFAULT_DATASET_PATH
    if df is None:
        df = load_url_dataset(dataset_path)
    info = dataset_info(dataset_path, df) if os.path.exists(dataset_path) else None

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    if config.max_train_samples and config.max_train_samples < len(train_df):
        train_df = train_df.sample(n=config.max_train_samples, random_state=config.seed)
    train_urls = train_df["url"].tolist()
    train_labels = train_df["label"].to_numpy(dtype=np.float32)
    val_urls = val_df["url"].tolist()
    val_labels = val_df["label"].to_numpy(dtype=int)

    vocab_path = vocab_path_for(models_dir)
    tok = build_or_load_vocab(train_urls, config.vocab_size, path=vocab_path, rebuild=rebuild_vocab)
    train_segments = _segments(train_urls)
    complete_train_texts = [compose_url_text(s) for s in train_segments]
    val_ids, val_mask = _encode_all(tok, [compose_url_text(s) for s in _segments(val_urls)], config.max_len)

    model = TransformerURLClassifier(
        vocab_size=max(config.vocab_size, max(tok.token_to_id.values()) + 1),
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        ff_dim=config.ff_dim,
        dropout=config.dropout,
        max_len=config.max_len,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    loss_fn = nn.BCEWithLogitsLoss()
    rng = np.random.RandomState(config.seed)
    labels_t = torch.tensor(train_labels, dtype=torch.float32)

    best_val_f1 = -1.0
    best_state = None
    history: List[Dict[str, object]] = []
    started = time.time()
    for epoch in range(1, config.epochs + 1):
        if variant == "robust":
            texts = augment_texts(train_segments, config.aug_max_rate, config.aug_representation, rng)
        else:
            texts = complete_train_texts
        ids, mask = _encode_all(tok, texts, config.max_len)
        order = torch.tensor(rng.permutation(len(ids)))
        model.train()
        total = 0.0
        steps = 0
        for i in range(0, len(order), config.batch_size):
            idx = order[i : i + config.batch_size]
            logits = model(ids[idx].to(device), mask[idx].to(device))
            loss = loss_fn(logits, labels_t[idx].to(device))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.item())
            steps += 1
        val = _evaluate(model, val_ids, val_mask, val_labels, device)
        record = {"epoch": epoch, "train_loss": total / max(steps, 1), "val_accuracy": val["accuracy"], "val_f1": val["f1"]}
        history.append(record)
        logger.info("url-classifier[{}] epoch {} loss={:.4f} val_acc={:.4f} val_f1={:.4f}", variant, epoch, record["train_loss"], val["accuracy"], val["f1"])
        print(f"[{variant}] epoch {epoch}/{config.epochs} loss={record['train_loss']:.4f} val_acc={val['accuracy']:.4f} val_f1={val['f1']:.4f}", flush=True)
        if val["f1"] > best_val_f1:
            best_val_f1 = float(val["f1"])
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    out_path = out_path or model_path_for(variant, models_dir)
    meta = {
        "variant": variant,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "train_seconds": round(time.time() - started, 1),
        "train_config": config.to_dict(),
        "n_train": int(len(train_urls)),
        "n_val": int(len(val_urls)),
        "vocab_path": os.path.relpath(vocab_path, PACKAGE_ROOT) if vocab_path.startswith(PACKAGE_ROOT) else vocab_path,
        "vocab_tokens": len(tok.token_to_id),
        "vocab_sha256": _file_sha256(vocab_path),
        "dataset_sha256": info.sha256 if info else None,
        "dataset_path": os.path.relpath(info.path, PACKAGE_ROOT) if info else None,
        "history": history,
        "best_val_f1": best_val_f1,
        "mitigation": "missing_information_augmentation" if variant == "robust" else "none",
    }
    save_url_classifier(model, out_path, extra_meta=meta)
    with open(os.path.splitext(out_path)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return {"path": out_path, **meta}


__all__ = ["MODELS_DIR", "VOCAB_PATH", "VARIANTS", "TrainConfig", "model_path_for", "vocab_path_for", "build_or_load_vocab", "augment_texts", "train_url_classifier"]
