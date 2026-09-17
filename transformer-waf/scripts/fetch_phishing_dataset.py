from __future__ import annotations

"""
Fetch (or convert) a labeled phishing/legitimate URL dataset into the
canonical data/phishing/urls.csv format (url, label, split).

Default source: PhiUSIIL Phishing URL Dataset (UCI ML Repository, CC BY 4.0).
  Prasad, A. & Chandra, S. (2023). PhiUSIIL: A diverse security profile
  empowered phishing URL detection framework based on similarity index and
  incremental learning. Computers & Security. https://doi.org/10.24432/C5J92R
Only the URL and label columns are used. In the source file label 1 means
legitimate; the canonical file uses label 1 = phishing.

Usage:
  PYTHONPATH=. python scripts/fetch_phishing_dataset.py                 # download PhiUSIIL
  PYTHONPATH=. python scripts/fetch_phishing_dataset.py --from-csv my.csv --url-col URL --label-col Label --positive bad
  PYTHONPATH=. python scripts/fetch_phishing_dataset.py --merge my_extra_urls.csv     # PhiUSIIL + your own url,label rows
"""

import argparse
import io
import os
import sys
import urllib.request
import zipfile

from src.phishing.dataset import DEFAULT_DATASET_PATH, prepare_url_dataset

PHIUSIIL_URL = "https://archive.ics.uci.edu/static/public/967/phiusiil+phishing+url+dataset.zip"
PHIUSIIL_CSV = "PhiUSIIL_Phishing_URL_Dataset.csv"


def download_phiusiil(dest_dir: str) -> str:
    os.makedirs(dest_dir, exist_ok=True)
    target = os.path.join(dest_dir, PHIUSIIL_CSV)
    if os.path.exists(target):
        print(f"using existing {target}")
        return target
    print(f"downloading {PHIUSIIL_URL} ...")
    with urllib.request.urlopen(PHIUSIIL_URL, timeout=120) as resp:  # noqa: S310 - fixed, documented source
        data = resp.read()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        zf.extract(PHIUSIIL_CSV, dest_dir)
    return target


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-csv", help="Use a local CSV instead of downloading PhiUSIIL")
    ap.add_argument("--url-col", default="URL")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--positive", default=None, help="Label value that denotes phishing (default: auto for PhiUSIIL)")
    ap.add_argument("--out", default=DEFAULT_DATASET_PATH)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--keep-raw", action="store_true", help="Keep the downloaded raw CSV")
    ap.add_argument("--merge", nargs="*", default=[], help="Extra CSVs with columns url,label (1=phishing/0=legitimate) to add before splitting")
    args = ap.parse_args()

    raw_dir = os.path.join(os.path.dirname(args.out) or ".", "raw")
    if args.from_csv:
        source = args.from_csv
        positive = args.positive
    else:
        source = download_phiusiil(raw_dir)
        # PhiUSIIL: label 0 = phishing, 1 = legitimate
        positive = args.positive if args.positive is not None else "0"

    info = prepare_url_dataset(
        source, out_path=args.out, url_col=args.url_col, label_col=args.label_col, positive_value=positive, seed=args.seed, merge_csvs=args.merge
    )
    print(f"wrote {info.path}")
    print(f"  total={info.n_total} phishing={info.n_phishing} legitimate={info.n_legitimate}")
    print(f"  splits={info.split_counts}")
    print(f"  sha256={info.sha256}")
    if not args.from_csv and not args.keep_raw:
        try:
            os.remove(source)
        except OSError as exc:
            print(f"could not remove raw file: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
