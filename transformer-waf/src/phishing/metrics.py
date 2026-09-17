"""
Classification metrics for the phishing experiments. Phishing is the positive
class (label 1).
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score


METRIC_KEYS = ("accuracy", "precision", "recall", "f1", "fpr", "fnr", "roc_auc")


def compute_metrics(y_true: Sequence[int], y_pred: Sequence[int], y_score: Optional[Sequence[float]] = None) -> Dict[str, object]:
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)
    if yt.shape != yp.shape:
        raise ValueError("y_true and y_pred must have the same length")
    n = int(len(yt))
    tp = int(((yt == 1) & (yp == 1)).sum())
    tn = int(((yt == 0) & (yp == 0)).sum())
    fp = int(((yt == 0) & (yp == 1)).sum())
    fn = int(((yt == 1) & (yp == 0)).sum())

    def _div(a: float, b: float) -> float:
        return float(a) / float(b) if b else 0.0

    precision = _div(tp, tp + fp)
    recall = _div(tp, tp + fn)
    f1 = _div(2 * precision * recall, precision + recall) if (precision + recall) else 0.0
    roc_auc: Optional[float] = None
    if y_score is not None and len(set(yt.tolist())) == 2:
        roc_auc = float(roc_auc_score(yt, np.asarray(y_score, dtype=float)))
    return {
        "n": n,
        "accuracy": _div(tp + tn, n),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": _div(fp, fp + tn),
        "fnr": _div(fn, fn + tp),
        "roc_auc": roc_auc,
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def metric_delta(reference: Dict[str, object], other: Dict[str, object]) -> Dict[str, Optional[float]]:
    """other - reference for each scalar metric (negative = degradation for
    accuracy/precision/recall/f1/roc_auc; positive = degradation for fpr/fnr)."""
    out: Dict[str, Optional[float]] = {}
    for key in METRIC_KEYS:
        a = reference.get(key)
        b = other.get(key)
        out[key] = (float(b) - float(a)) if (a is not None and b is not None) else None
    return out


__all__ = ["METRIC_KEYS", "compute_metrics", "metric_delta"]
