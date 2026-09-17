@echo off
REM One-command setup + start for Windows (run from anywhere; needs Python 3.11+ on PATH).
cd /d "%~dp0\.."
if not exist venv (
  echo [1/4] creating virtual environment
  python -m venv venv || exit /b 1
)
call venv\Scripts\activate.bat
echo [2/4] installing dependencies (first time downloads PyTorch, ~200 MB)
python -m pip install -q --upgrade pip
pip install -q -r requirements.txt || exit /b 1
set PYTHONPATH=.
echo [3/4] checking model artifacts
if not exist models\checkpoints\best.pt (
  echo     WAF model missing - training
  python scripts\prepare_waf_data.py
  python scripts\train_quick.py --epochs 40 --batch 64 --vocab 5000 --embed 128 --heads 4 --layers 3 --ff 256 --maxlen 128 --lr 5e-4
  python scripts\calibrate_threshold.py --benign-percentile 0.99 --write
)
if not exist data\phishing\urls.csv (
  echo     URL dataset missing - downloading PhiUSIIL
  python scripts\fetch_phishing_dataset.py
)
if not exist models\phishing\url_classifier_robust.pt (
  echo     URL classifiers missing - training both variants (about 20 minutes)
  python scripts\train_url_classifier.py --variant both --epochs 3 --rebuild-vocab
)
echo [4/4] starting the service on http://localhost:8000  (API key: dev-key, Ctrl+C to stop)
python -m uvicorn src.api.detection_api:app --host 0.0.0.0 --port 8000
