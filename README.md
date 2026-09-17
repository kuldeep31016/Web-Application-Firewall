# Transformer WAF + Robust Phishing URL Detection Under Incomplete Information

A Transformer-based Web Application Firewall (built for Smart India Hackathon)
extended into a research system for the question:

> How does missing feature information affect the ability of a machine-learning
> model to detect phishing URLs, and can an appropriate missing-information
> handling strategy enable the model to continue detecting phishing URLs reliably?

Everything lives in [`transformer-waf/`](transformer-waf/) and runs as **one**
FastAPI service on port 8000:

* **WAF anomaly detection** (original): a Transformer trained on benign HTTP
  requests scores how unusual a request is (`POST /detect`, batch, history,
  replay, threshold, retraining).
* **Phishing URL classification** (new): the same Transformer encoder with a
  classification head, trained on the labeled PhiUSIIL URL dataset
  (`POST /detect/url`).
* **Robustness experiments** (new): URL segments (scheme, subdomain, domain,
  TLD, path, query, fragment) are removed in a controlled, seeded way; three
  handling strategies — blank, explicit `[MISSING]` marker, and a model trained
  with incomplete examples — are compared on the same test URLs with accuracy,
  precision, recall, F1, FPR, FNR and ROC AUC. Results are stored and charted.

## Quick start

```bash
cd transformer-waf
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# WAF anomaly model
PYTHONPATH=. python scripts/prepare_waf_data.py
PYTHONPATH=. python scripts/train_quick.py --epochs 40 --batch 64 --vocab 5000 --embed 128 --heads 4 --layers 3 --ff 256 --maxlen 128 --lr 5e-4
PYTHONPATH=. python scripts/calibrate_threshold.py --benign-percentile 0.99 --write

# Phishing URL classifiers (downloads PhiUSIIL, ~15 MB)
PYTHONPATH=. python scripts/fetch_phishing_dataset.py
PYTHONPATH=. python scripts/train_url_classifier.py --variant both --epochs 3 --rebuild-vocab

# Serve
PYTHONPATH=. uvicorn src.api.detection_api:app --host 0.0.0.0 --port 8000
```

Then open <http://localhost:8000/> (overview), <http://localhost:8000/analyze>
(analyzer), <http://localhost:8000/research> (experiments),
<http://localhost:8000/history> (persistence) and <http://localhost:8000/docs>
(API). Default API key: `dev-key`.

Hand-over documents (PDF + Markdown): [`transformer-waf/docs/`](transformer-waf/docs/) —
HANDOVER, DEMO_SCRIPT (5-minute demo, click by click) and EXAMPLES (verified test URLs).

Full documentation — architecture, what "features" mean for a sequence model,
the missing-information simulation, experiment methodology, measured results,
API reference, tests and limitations — is in
[`transformer-waf/README.md`](transformer-waf/README.md).

## Tests

```bash
cd transformer-waf && PYTHONPATH=. python -m pytest -q tests
```

## License / credits

Built for Smart India Hackathon — Team SecuraFormer. Phishing URL data:
PhiUSIIL Phishing URL Dataset (UCI ML Repository, CC BY 4.0).
