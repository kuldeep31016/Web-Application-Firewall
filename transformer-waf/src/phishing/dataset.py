"""
Labeled URL dataset interface for the phishing robustness experiments.

The canonical on-disk format is a CSV with three columns:

    url    - the raw URL string
    label  - 1 = phishing, 0 = legitimate
    split  - "train", "val" or "test" (assigned once, deterministically)

``prepare_url_dataset`` converts any source CSV with a URL column and a label
column into this format (de-duplicating URLs and assigning a stratified split
with a fixed seed), so the experiments never train and evaluate on the same
URL and every run uses the same split.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_DATASET_PATH = os.path.join(PACKAGE_ROOT, "data", "phishing", "urls.csv")

POSITIVE_LABELS = {"1", "phishing", "phish", "bad", "malicious", "true", "yes"}
NEGATIVE_LABELS = {"0", "legitimate", "legit", "good", "benign", "false", "no"}
SPLITS = ("train", "val", "test")


class DatasetError(ValueError):
    """Raised when a dataset file is missing or malformed."""


@dataclass
class DatasetInfo:
    path: str
    n_total: int
    n_phishing: int
    n_legitimate: int
    split_counts: Dict[str, int]
    sha256: str
    label_mapping: str = "1 = phishing, 0 = legitimate"

    def to_dict(self) -> Dict[str, object]:
        return {
            "path": self.path,
            "n_total": self.n_total,
            "n_phishing": self.n_phishing,
            "n_legitimate": self.n_legitimate,
            "split_counts": self.split_counts,
            "sha256": self.sha256,
            "label_mapping": self.label_mapping,
        }


def _normalize_label(value: object, positive_value: Optional[str] = None) -> int:
    text = str(value).strip().lower()
    if positive_value is not None:
        return 1 if text == str(positive_value).strip().lower() else 0
    if text in POSITIVE_LABELS:
        return 1
    if text in NEGATIVE_LABELS:
        return 0
    raise DatasetError(f"Unrecognised label value {value!r}; pass positive_value explicitly")


def assign_splits(labels: Sequence[int], seed: int, val_frac: float = 0.1, test_frac: float = 0.2) -> List[str]:
    """Stratified train/val/test assignment with a fixed seed."""
    labels_arr = np.asarray(labels)
    rng = np.random.RandomState(seed)
    out = np.empty(len(labels_arr), dtype=object)
    for cls in np.unique(labels_arr):
        idx = np.where(labels_arr == cls)[0]
        rng.shuffle(idx)
        n_test = int(round(len(idx) * test_frac))
        n_val = int(round(len(idx) * val_frac))
        out[idx[:n_test]] = "test"
        out[idx[n_test : n_test + n_val]] = "val"
        out[idx[n_test + n_val :]] = "train"
    return out.tolist()


def _read_source(source_csv: str, url_col: str, label_col: str, positive_value: Optional[str], max_url_length: int) -> pd.DataFrame:
    if not os.path.exists(source_csv):
        raise DatasetError(f"Source CSV not found: {source_csv}")
    df = pd.read_csv(source_csv, usecols=[url_col, label_col], encoding="utf-8-sig", dtype=str)
    df = df.rename(columns={url_col: "url", label_col: "label"})
    df["url"] = df["url"].astype(str).str.strip()
    df = df[(df["url"] != "") & (df["url"].str.len() <= max_url_length)]
    df["label"] = df["label"].map(lambda v: _normalize_label(v, positive_value))
    return df


def prepare_url_dataset(
    source_csv: str,
    out_path: str = DEFAULT_DATASET_PATH,
    url_col: str = "URL",
    label_col: str = "label",
    positive_value: Optional[str] = None,
    seed: int = 42,
    max_url_length: int = 2048,
    merge_csvs: Optional[Sequence[str]] = None,
) -> DatasetInfo:
    """Convert a source CSV (plus optional extra CSVs in the canonical
    ``url,label`` form) to the canonical url/label/split format."""
    df = _read_source(source_csv, url_col, label_col, positive_value, max_url_length)
    for extra in merge_csvs or []:
        df = pd.concat([df, _read_source(extra, "url", "label", None, max_url_length)], ignore_index=True)
    df = df.drop_duplicates(subset="url").reset_index(drop=True)
    df["split"] = assign_splits(df["label"].tolist(), seed=seed)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df.to_csv(out_path, index=False)
    return dataset_info(out_path, df)


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_url_dataset(path: str = DEFAULT_DATASET_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise DatasetError(
            f"Dataset not found at {path}. Run `PYTHONPATH=. python scripts/fetch_phishing_dataset.py` "
            "or prepare your own CSV with `--from-csv`."
        )
    df = pd.read_csv(path, dtype={"url": str, "label": int, "split": str})
    missing = {"url", "label", "split"} - set(df.columns)
    if missing:
        raise DatasetError(f"Dataset {path} is missing columns: {sorted(missing)}")
    if not set(df["label"].unique()) <= {0, 1}:
        raise DatasetError("Dataset labels must be 0 (legitimate) or 1 (phishing)")
    if not set(df["split"].unique()) <= set(SPLITS):
        raise DatasetError(f"Dataset split values must be one of {SPLITS}")
    df["url"] = df["url"].fillna("").astype(str)
    return df


def dataset_info(path: str, df: Optional[pd.DataFrame] = None) -> DatasetInfo:
    if df is None:
        df = load_url_dataset(path)
    counts = df["split"].value_counts().to_dict()
    return DatasetInfo(
        path=path,
        n_total=int(len(df)),
        n_phishing=int((df["label"] == 1).sum()),
        n_legitimate=int((df["label"] == 0).sum()),
        split_counts={s: int(counts.get(s, 0)) for s in SPLITS},
        sha256=_file_sha256(path),
    )


def sample_split(df: pd.DataFrame, split: str, n: Optional[int], seed: int) -> pd.DataFrame:
    """Return a deterministic stratified subsample of one split (or the whole split)."""
    part = df[df["split"] == split]
    if n is None or n >= len(part):
        return part.reset_index(drop=True)
    rng = np.random.RandomState(seed)
    pieces = []
    for cls, grp in part.groupby("label"):
        k = int(round(n * len(grp) / len(part)))
        idx = rng.choice(len(grp), size=min(k, len(grp)), replace=False)
        pieces.append(grp.iloc[np.sort(idx)])
    return pd.concat(pieces).reset_index(drop=True)


__all__ = [
    "DEFAULT_DATASET_PATH",
    "DatasetError",
    "DatasetInfo",
    "assign_splits",
    "prepare_url_dataset",
    "load_url_dataset",
    "dataset_info",
    "sample_split",
]
