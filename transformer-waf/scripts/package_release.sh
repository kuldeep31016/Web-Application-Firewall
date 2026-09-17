#!/usr/bin/env bash
# Build a self-contained zip of the project INCLUDING the trained models and the
# prepared dataset, so the recipient can run it without training anything.
# Excludes the virtualenv, caches, raw downloads and old encoded training dumps.
#
#   ./scripts/package_release.sh            -> ../transformer-waf-release-YYYYMMDD.zip
set -euo pipefail
cd "$(dirname "$0")/.."
NAME="transformer-waf-release-$(date +%Y%m%d)"
OUT="$(cd .. && pwd)/${NAME}.zip"

for f in models/checkpoints/best.pt models/checkpoints/vocab.json models/phishing/url_classifier_baseline.pt \
         models/phishing/url_classifier_robust.pt models/phishing/vocab.json data/phishing/urls.csv; do
  [ -f "$f" ] || { echo "missing artifact: $f (train / fetch it first)"; exit 1; }
done

rm -f "$OUT"
cd ..
zip -qr "$OUT" transformer-waf \
  -x "transformer-waf/venv/*" "transformer-waf/.venv/*" "*/__pycache__/*" "*.pyc" "*/.pytest_cache/*" \
     "transformer-waf/data/phishing/raw/*" \
     "transformer-waf/data/train/_backup_train.jsonl" "transformer-waf/data/train/app1.jsonl" \
     "transformer-waf/data/train/app2.jsonl" "transformer-waf/data/train/app3.jsonl" \
     "transformer-waf/logs/*.log" "transformer-waf/logs/detection_logs/*" "transformer-waf/logs/nginx/*" \
     "transformer-waf/.demo_pids" "*/.DS_Store"
echo "wrote $OUT  ($(du -h "$OUT" | cut -f1))"
echo "recipient runs:  unzip, then  ./transformer-waf/scripts/setup_and_run.sh   (Windows: scripts\\setup_and_run.bat)"
