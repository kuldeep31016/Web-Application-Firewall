#!/usr/bin/env bash
# One-command setup + start for a fresh machine (macOS / Linux).
# Creates a virtual environment, installs dependencies, checks that the trained
# models are present (trains them if not) and starts the web service.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-python3}"
PORT="${PORT:-8000}"

if [ ! -d venv ]; then
  echo "[1/4] creating virtual environment"
  "$PY" -m venv venv
fi
# shellcheck disable=SC1091
source venv/bin/activate
echo "[2/4] installing dependencies (first time downloads PyTorch, ~200 MB)"
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo "[3/4] checking model artifacts"
if [ ! -f models/checkpoints/best.pt ]; then
  echo "    WAF model missing -> training (about 3 minutes)"
  PYTHONPATH=. python scripts/prepare_waf_data.py
  PYTHONPATH=. python scripts/train_quick.py --epochs 40 --batch 64 --vocab 5000 --embed 128 --heads 4 --layers 3 --ff 256 --maxlen 128 --lr 5e-4
  PYTHONPATH=. python scripts/calibrate_threshold.py --benign-percentile 0.99 --write
fi
if [ ! -f data/phishing/urls.csv ]; then
  echo "    URL dataset missing -> downloading PhiUSIIL (about 15 MB)"
  PYTHONPATH=. python scripts/fetch_phishing_dataset.py
fi
if [ ! -f models/phishing/url_classifier_baseline.pt ] || [ ! -f models/phishing/url_classifier_robust.pt ]; then
  echo "    URL classifiers missing -> training both variants (about 20 minutes on CPU)"
  PYTHONPATH=. python scripts/train_url_classifier.py --variant both --epochs 3 --rebuild-vocab
fi

echo "[4/4] starting the service on http://localhost:${PORT}  (API key: dev-key, Ctrl+C to stop)"
PYTHONPATH=. exec python -m uvicorn src.api.detection_api:app --host 0.0.0.0 --port "$PORT"
