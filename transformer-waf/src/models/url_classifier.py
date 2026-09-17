"""
Supervised phishing/legitimate URL classifier built on the existing WAF
Transformer encoder.

The anomaly-detecting ``WAFTransformer`` is trained unsupervised on benign
traffic and therefore cannot say whether a URL is *phishing*; it can only say
whether a request is *unusual*. To answer the research question a labeled
decision is needed, so this module adds a classification head on top of the
same embedding + Transformer encoder stack (``WAFTransformer.encode``) and
trains it with binary cross-entropy on labeled URLs. Tokenisation is the same
``HTTPRequestTokenizer`` used by the WAF.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from ..preprocessing.tokenizer import HTTPRequestTokenizer
from .transformer_model import WAFTransformer


class TransformerURLClassifier(nn.Module):
    """WAF Transformer encoder + masked mean pooling + linear head (1 logit)."""

    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int, num_layers: int, ff_dim: int = 256, dropout: float = 0.1, max_len: int = 64) -> None:
        super().__init__()
        self.backbone = WAFTransformer(
            vocab_size=vocab_size, embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers, ff_dim=ff_dim, dropout=dropout, max_len=max_len
        )
        # The reconstruction decoder is not used for classification.
        self.backbone.decoder = nn.Identity()
        self.backbone.proj = nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(embed_dim, 1)
        self.meta: Dict[str, object] = {
            "vocab_size": vocab_size,
            "embed_dim": embed_dim,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "ff_dim": ff_dim,
            "dropout": dropout,
            "max_len": max_len,
        }

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone.encode(input_ids, attention_mask, causal=False)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        return self.head(self.dropout(pooled)).squeeze(-1)

    @torch.no_grad()
    def predict_proba(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(input_ids, attention_mask))


def save_url_classifier(model: TransformerURLClassifier, path: str, extra_meta: Optional[Dict[str, object]] = None) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    meta = dict(model.meta)
    if extra_meta:
        meta.update(extra_meta)
    torch.save({"model": model.state_dict(), "meta": meta}, path)


def load_url_classifier(path: str, device: Optional[torch.device] = None) -> TransformerURLClassifier:
    device = device or torch.device("cpu")
    ckpt = torch.load(path, map_location=device)
    meta = ckpt["meta"]
    model = TransformerURLClassifier(
        vocab_size=int(meta["vocab_size"]),
        embed_dim=int(meta["embed_dim"]),
        num_heads=int(meta["num_heads"]),
        num_layers=int(meta["num_layers"]),
        ff_dim=int(meta.get("ff_dim", 256)),
        dropout=float(meta.get("dropout", 0.1)),
        max_len=int(meta["max_len"]),
    )
    model.load_state_dict(ckpt["model"])
    model.meta = dict(meta)
    model.to(device)
    model.eval()
    return model


class URLClassifierEngine:
    """Holds a loaded classifier + tokenizer and scores URL texts in batches."""

    def __init__(self, model: TransformerURLClassifier, tokenizer: HTTPRequestTokenizer, threshold: float = 0.5, device: Optional[torch.device] = None) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.threshold = float(threshold)
        self.device = device or torch.device("cpu")
        self.max_len = int(model.meta["max_len"])

    def encode(self, texts: List[str]):
        enc = [self.tokenizer.encode(t, max_length=self.max_len) for t in texts]
        ids = torch.tensor([e["input_ids"] for e in enc], dtype=torch.long, device=self.device)
        mask = torch.tensor([e["attention_mask"] for e in enc], dtype=torch.long, device=self.device)
        return ids, mask

    @torch.no_grad()
    def score(self, texts: List[str], batch_size: int = 256) -> List[float]:
        """Phishing probability for each text."""
        out: List[float] = []
        for i in range(0, len(texts), batch_size):
            ids, mask = self.encode(texts[i : i + batch_size])
            out.extend(self.model.predict_proba(ids, mask).tolist())
        return out

    def predict(self, texts: List[str], batch_size: int = 256) -> List[Dict[str, object]]:
        probs = self.score(texts, batch_size=batch_size)
        return [{"phishing_probability": p, "is_phishing": p >= self.threshold} for p in probs]


__all__ = ["TransformerURLClassifier", "URLClassifierEngine", "save_url_classifier", "load_url_classifier"]
