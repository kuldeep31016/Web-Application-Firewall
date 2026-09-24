# Transformer WAF + Robust Phishing URL Detection Under Incomplete Information

One FastAPI service, two Transformer models, one question added on top of the
original WAF:

> How does missing feature information affect the ability of a machine-learning
> model to detect phishing URLs, and can a missing-information handling strategy
> let it keep detecting phishing URLs reliably?

| Component | What it is | What it answers |
|---|---|---|
| **WAF anomaly model** (original) | Transformer autoencoder trained *only on benign requests*; anomaly score = mean next-token negative log-likelihood | "Is this HTTP request unlike normal traffic?" |
| **URL classifier** (new) | The *same* Transformer encoder (`WAFTransformer.encode`) with a classification head, trained *supervised* on labeled phishing/legitimate URLs | "Is this URL phishing or legitimate?" |
| **Research module** (new) | Controlled removal of URL information, three handling strategies, metrics, persistence, UI | "What happens when information is missing, and does the mitigation help?" |

The anomaly model is **not** a phishing classifier and the UI never presents it
as one. Both are reported side by side on the Analyze page.

Quick links (local): [Overview](http://localhost:8000/) · [Analyze](http://localhost:8000/analyze) · [Research](http://localhost:8000/research) · [History](http://localhost:8000/history) · [Analytics](http://localhost:8000/analytics) ([guide](docs/ANALYTICS_GUIDE.pdf)) · [API docs](http://localhost:8000/docs)

Hand-over documents (Markdown + PDF) are in [`docs/`](docs/): **HANDOVER** (how it all fits together), **DEMO_SCRIPT** (5-minute click-by-click demo) and **EXAMPLES** (verified test URLs and expected results).

---

## 0. Project brief coverage

The brief — *Robust Phishing URL Detection Under Incomplete Feature Information Using Machine Learning* — maps onto the code as follows. "Present" = implemented in the first build; "added" = added when the brief was formalised.

| Brief requirement | Where | Status |
|---|---|---|
| Phishing URL dataset + preprocessing, leak-free split, fixed seed | `scripts/fetch_phishing_dataset.py`, `src/phishing/dataset.py` (`--merge` for extra `url,label` rows) | present (+ `--merge` added) |
| Baseline classifier trained on complete information, metrics | `src/models/url_classifier.py`, `src/phishing/training.py` (baseline variant) | present |
| Controlled missing-information generation (0–50 %, fixed features) | `src/phishing/url_features.py`, `src/phishing/missingness.py` | present |
| Baseline evaluated under missing information, same test set | `src/phishing/experiments.py` → `run_missing_information` | present |
| Mitigation: train with incomplete examples (robust model) | `src/phishing/training.py` (`robust` variant, `augment_texts`) | present |
| Feature-dependency analysis | `run_feature_impact` | present |
| Comparison table + accuracy-vs-missing graph (+ F1 option) | Research page, Overview page, `scripts/run_robustness_experiments.py` | present |
| URL analyzer: prediction, probability, segments, information availability, model used | `/analyze` | present (+ availability bar added) |
| Interactive missing-information demo (complete → missing → mitigated; 0/10/30/50 % levels) | Overview "Live analysis"; `/analyze` **Run experiment** and **Missing-level tests** | present (+ both buttons added) |
| Overview page: hero, dynamic metrics, 4-step method, research visual | `/` | added (this revision) |
| Research page: problem, question, dataset, model, method, mitigation, metrics, results, conclusion | `/research#overview` | added |
| Results page: baseline cards, full metric table, feature-dependency table | `/research#results` (nav **Results**) | added |
| Reproducibility: seeds, split, checkpoints, stored config + metrics | `models/phishing/*.json`, `logs/detections.db` (`experiments`) | present |
| cURL / Swagger examples for request and URL scoring | `/docs` (pre-filled examples), §9 below | added |

Out of scope by design (not implemented): threat-intelligence feeds, WHOIS/DNS, browser extension, malware scanning, authentication, additional ML algorithms. The WAF request anomaly model is kept as the original component; it is not part of the research question.

## 1. Setup

```bash
cd transformer-waf
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Python 3.11+ (developed on 3.14, PyTorch 2.x, CPU is enough).

## 2. Train the WAF anomaly model (existing functionality)

```bash
# benign-only training set + vocabulary + held-out calibration sets from data/training/*.jsonl
PYTHONPATH=. python scripts/prepare_waf_data.py
# small Transformer (about 3 min on CPU for 40 epochs)
PYTHONPATH=. python scripts/train_quick.py --epochs 40 --batch 64 --vocab 5000 --embed 128 --heads 4 --layers 3 --ff 256 --maxlen 128 --lr 5e-4
# measure TPR/FPR on held-out benign vs attack requests and write detection.threshold to config.yaml
PYTHONPATH=. python scripts/calibrate_threshold.py --benign-percentile 0.99 --write
```

`calibrate_threshold.py` prints the score distributions and the true-positive /
false-positive rate at several thresholds so the chosen value is a measured one.

## 3. Get a labeled URL dataset (new)

```bash
# PhiUSIIL Phishing URL Dataset (UCI, CC BY 4.0): 235,795 URLs -> data/phishing/urls.csv (url,label,split)
PYTHONPATH=. python scripts/fetch_phishing_dataset.py
# or your own CSV
PYTHONPATH=. python scripts/fetch_phishing_dataset.py --from-csv my.csv --url-col URL --label-col Label --positive bad
```

The canonical file has `url`, `label` (**1 = phishing, 0 = legitimate**) and a
deterministic stratified `split` (70 % train / 10 % val / 20 % test, seed 42)
assigned once after de-duplicating URLs. Training only ever reads `train`
(+ `val` for model selection); experiments only ever read `test`. The file's
SHA-256 is recorded with every model and every experiment.

## 4. Train the URL classifiers (new)

```bash
PYTHONPATH=. python scripts/train_url_classifier.py --variant both --epochs 3 --rebuild-vocab
```

Two checkpoints in `models/phishing/` share one vocabulary and one split:

* `url_classifier_baseline.pt` — trained on complete URLs.
* `url_classifier_robust.pt` — **the mitigation**: trained with
  *missing-information augmentation*. In every epoch each training URL gets a
  fresh random subset of its segments replaced by `[MISSING]` (per-URL rate
  drawn from U(0, 0.5)), so the model learns to decide from whatever remains.

Each checkpoint stores its training config, seed, dataset hash, vocabulary hash
and validation history (`models/phishing/*.json`).

## 5. Run the service

```bash
PYTHONPATH=. uvicorn src.api.detection_api:app --host 0.0.0.0 --port 8000
# optional: update/retrain API
PYTHONPATH=. uvicorn src.api.update_api:app --host 0.0.0.0 --port 8001
# or everything (APIs + sample apps): ./scripts/start_demo_services.sh
```

Every endpoint requires `X-API-Key` (default `dev-key`, override with `WAF_API_KEY`).

## 5a. Running it on another machine (no training needed)

Trained models and the prepared dataset are not in Git, so the fastest way to
hand the project over is a release zip that already contains them:

```bash
./scripts/package_release.sh        # -> ../transformer-waf-release-YYYYMMDD.zip (about 22 MB)
```

The recipient needs only Python 3.11+ and an internet connection for `pip`:

```bash
unzip transformer-waf-release-*.zip
./transformer-waf/scripts/setup_and_run.sh      # macOS / Linux
transformer-waf\scripts\setup_and_run.bat       # Windows
```

The script creates `venv/`, installs `requirements.txt` (PyTorch is the only
large download), verifies that the models are present (it trains or downloads
anything missing, so it also works from a bare `git clone`), and starts the
service on http://localhost:8000. Stored experiments in `logs/detections.db`
travel with the zip, so the Research page shows results immediately.

Docker is the alternative: `docker compose up -d waf-api` builds the same
service (copy the models/dataset in first, or mount `./models` and `./data`
as the compose file already does).

## 6. What "features" mean here

The Transformer does not consume a tabular feature vector — it consumes a token
sequence built from the URL. The units of information that can genuinely be
*unavailable* are therefore the structural **segments** the sequence is composed
from (`src/phishing/url_features.py`):

| Segment | Example |
|---|---|
| `scheme` | `https` |
| `subdomain` | `login.secure` |
| `domain` | `paypal-verify` |
| `tld` | `co.uk` |
| `path` | `/account/verify.php` |
| `query` | `id=1&ref=mail` |
| `fragment` | `top` |

Lexical statistics such as "URL length" are *not* separate model inputs, so
they are not maskable features; the UI shows a few of them as descriptive
information only.

## 7. Missing-information simulation

`src/phishing/missingness.py` removes segments **before tokenisation** and
composes the model text only from what remains — nothing downstream can
reconstruct a removed segment.

* **Fixed features**: a chosen set of segments is unavailable for every URL
  (single-feature and multi-feature conditions, feature-dependency analysis).
* **Missing rate** `p` ∈ {0, 10, 20, 30, 40, 50 %}: each segment of each URL is
  independently unavailable with probability `p` (Bernoulli, seeded). The
  realised fraction is reported next to the nominal one.

Three handling strategies:

| Strategy | Representation | Model | Meaning |
|---|---|---|---|
| `none` | segment blanked | baseline | information simply absent; nothing done about it |
| `missing_token` | `[MISSING]` marker | baseline | explicit missing representation without retraining |
| `augmented_training` | `[MISSING]` marker | robust | **the mitigation**: model trained with incomplete examples |

Example (`scheme` and `domain` unavailable):

```
original        https://login.secure-paypal.co.uk/account/verify.php?id=1
none            login.co.uk/account/verify.php?id=1
missing_token   [MISSING]://login.[MISSING].co.uk/account/verify.php?id=1
```

Why this strategy and not "replace with the average"? The model has no numeric
columns to average. An explicit marker plus training with incomplete examples is
the sequence-model analogue of the research guide's second suggestion
("deliberately train the model using incomplete examples").

## 8. Experiments

Every condition in one experiment is evaluated on the **same** stratified sample
of the **test** split with the **same** seeded missingness draw, so differences
between strategies are attributable to the strategy alone.

| Experiment | Endpoint / script | Conditions |
|---|---|---|
| Complete baseline | part of every run | rate 0 %, no fixed features, baseline model |
| One / multiple features missing | `features=[...]` | fixed segments removed for every URL |
| Missing-rate sweep | `rates=[0,.1,…,.5]` | per-segment Bernoulli removal |
| Missing + mitigation | strategy `augmented_training` | robust model on identical masked inputs |
| Feature dependency | `/experiment/feature-impact` | each segment (and optional combinations) removed in turn |

Metrics per condition: accuracy, precision, recall, F1, FPR, FNR, ROC AUC,
confusion matrix, degradation vs. the complete reference (Δ), and for the
mitigated condition the recovery vs. `none` at the same rate.

```bash
# CLI (prints Markdown tables, stores results, optional JSON)
PYTHONPATH=. python scripts/run_robustness_experiments.py --sample-size 10000 --seed 42 --out results.json
```

or use the **Research** page, or the API:

```bash
curl -s -X POST localhost:8000/experiment/missing-information -H "X-API-Key: dev-key" -H "Content-Type: application/json" \
  -d '{"rates":[0,0.1,0.2,0.3,0.4,0.5],"features":[],"sample_size":5000,"seed":42}'
curl -s -X POST localhost:8000/experiment/feature-impact -H "X-API-Key: dev-key" -H "Content-Type: application/json" -d '{"sample_size":5000}'
curl -s localhost:8000/experiment/results -H "X-API-Key: dev-key"
```

Results are persisted in `logs/detections.db` (`experiments`,
`experiment_results`) with experiment id, timestamp, seed, dataset path + hash,
evaluation split, sample size, configuration, model provenance and duration —
enough to re-run any experiment exactly.

### Measured results

All numbers below were produced by
`scripts/run_robustness_experiments.py --sample-size 0 --seed 42` on the
**whole PhiUSIIL test split (47,072 URLs; 20,102 phishing / 26,970 legitimate)**
with the checkpoints trained by the commands above (3 epochs each, seed 42).
They are stored in `logs/detections.db` and reproduced by re-running the
command. Re-training will change them slightly; nothing here is typed in by hand.

**Missing-information sweep** (Δ = change in accuracy vs. the complete reference)

| Condition | Strategy | Missing | Accuracy | Precision | Recall | F1 | FPR | FNR | ROC AUC | Δ acc |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| complete | none (baseline model) | 0 % | 99.80 % | 100.00 % | 99.53 % | 99.76 % | 0.00 % | 0.47 % | 0.9986 | ref |
| complete | augmented_training (robust model) | 0 % | 99.81 % | 100.00 % | 99.55 % | 99.77 % | 0.00 % | 0.45 % | 0.9988 | +0.01 pp |
| incomplete | none | 10 % | 92.63 % | 87.71 % | 96.24 % | 91.78 % | 10.06 % | 3.76 % | 0.9549 | −7.16 pp |
| incomplete | missing_token | 10 % | 87.79 % | 79.15 % | 96.94 % | 87.15 % | 19.03 % | 3.06 % | 0.9861 | −12.01 pp |
| **mitigated** | augmented_training | 10 % | **98.58 %** | 99.89 % | 96.77 % | 98.31 % | 0.08 % | 3.23 % | 0.9973 | −1.22 pp |
| incomplete | none | 20 % | 86.04 % | 78.10 % | 93.52 % | 85.12 % | 19.54 % | 6.48 % | 0.9093 | −13.76 pp |
| incomplete | missing_token | 20 % | 77.46 % | 66.45 % | 95.41 % | 78.34 % | 35.91 % | 4.59 % | 0.9613 | −22.33 pp |
| **mitigated** | augmented_training | 20 % | **97.11 %** | 99.76 % | 93.46 % | 96.51 % | 0.17 % | 6.54 % | 0.9930 | −2.69 pp |
| incomplete | none | 30 % | 79.71 % | 70.35 % | 90.75 % | 79.25 % | 28.51 % | 9.25 % | 0.8641 | −20.08 pp |
| incomplete | missing_token | 30 % | 68.54 % | 58.07 % | 94.69 % | 71.99 % | 50.95 % | 5.31 % | 0.9305 | −31.26 pp |
| **mitigated** | augmented_training | 30 % | **95.16 %** | 99.60 % | 89.03 % | 94.02 % | 0.27 % | 10.97 % | 0.9849 | −4.63 pp |
| incomplete | none | 40 % | 74.06 % | 64.33 % | 88.12 % | 74.37 % | 36.42 % | 11.88 % | 0.8205 | −25.74 pp |
| incomplete | missing_token | 40 % | 61.03 % | 52.42 % | 94.75 % | 67.49 % | 64.11 % | 5.25 % | 0.8991 | −38.77 pp |
| **mitigated** | augmented_training | 40 % | **92.84 %** | 99.39 % | 83.74 % | 90.90 % | 0.39 % | 16.26 % | 0.9719 | −6.96 pp |
| incomplete | none | 50 % | 69.14 % | 59.83 % | 84.36 % | 70.01 % | 42.21 % | 15.64 % | 0.7782 | −30.66 pp |
| incomplete | missing_token | 50 % | 55.03 % | 48.65 % | 95.45 % | 64.45 % | 75.11 % | 4.55 % | 0.8647 | −44.77 pp |
| **mitigated** | augmented_training | 50 % | **89.81 %** | 99.17 % | 76.78 % | 86.55 % | 0.48 % | 23.22 % | 0.9512 | −9.98 pp |

"Missing 30 %" means every segment of every URL was independently unavailable
with probability 0.3 (realised 29.96 %). The same masked inputs were given to
every strategy.

**Feature dependency** (one segment removed for every test URL, baseline model,
strategy `none`; "present" = share of test URLs where the segment is non-empty)

| Removed | Present | Accuracy | F1 | Δ acc | Δ F1 | Δ acc with mitigation |
|---|---:|---:|---:|---:|---:|---:|
| subdomain | 93.4 % | 42.71 % | 59.85 % | −57.09 pp | −39.91 pp | −0.33 pp |
| scheme | 100 % | 84.56 % | 77.94 % | −15.24 pp | −21.82 pp | −9.98 pp |
| path | 25.5 % | 99.21 % | 99.06 % | −0.59 pp | −0.70 pp | −0.43 pp |
| tld | 99.7 % | 99.73 % | 99.68 % | −0.07 pp | −0.08 pp | −0.01 pp |
| domain | 100 % | 99.77 % | 99.73 % | −0.03 pp | −0.03 pp | −0.01 pp |
| query | 2.6 % | 99.80 % | 99.76 % | +0.00 pp | +0.00 pp | +0.01 pp |
| fragment | 0.4 % | 99.80 % | 99.76 % | +0.00 pp | +0.00 pp | +0.01 pp |

**What the measurements say**

1. *Missing information degrades detection substantially.* With no handling,
   accuracy falls monotonically from 99.80 % to 69.14 % as the missing rate
   rises to 50 %, driven mainly by false positives (FPR 0 % → 42 %).
2. *An explicit marker alone does not help a model that never saw it.* The
   `[MISSING]` token without retraining is worse than blanking at every level
   (55.03 % at 50 %), although ROC AUC stays higher — the marker shifts scores
   rather than destroying ranking. Removing only `path` with the marker costs
   57 pp because no legitimate test URL has a path (0 %, vs. 59.7 % of phishing
   URLs), so appending `/[MISSING]` makes every legitimate URL look like a
   path-bearing phishing URL.
3. *Training with incomplete examples recovers most of the loss.* The robust
   model keeps 98.58 % / 95.16 % / 89.81 % accuracy at 10 % / 30 % / 50 %
   missing — a recovery of +5.95 / +15.45 / +20.67 pp over no handling — with
   FPR below 0.5 % throughout, and it is not worse with complete information
   (99.81 % vs 99.80 %). Its remaining loss is recall: at 50 % missing it misses
   23 % of phishing URLs.
4. *The baseline depends on two shallow cues.* Removing `subdomain` (mostly
   `www`) or `scheme` (`https`) is what breaks it; `domain` and `tld` are almost
   irrelevant on this dataset. The reason is in the data: in the test split
   100 % of legitimate URLs use `https` (phishing: 48.5 %), 93.7 % start with
   `www` (phishing: 30.1 %) and 0 % have a path (phishing: 59.7 %). The robust
   model no longer needs `subdomain` (−0.33 pp) but still leans on `scheme`
   (−9.98 pp). This is a property of PhiUSIIL that the experiment exposes, not
   a general truth about URLs.

## 9. API

Existing endpoints are unchanged: `POST /detect`, `POST /detect/batch`,
`GET /logs`, `POST /replay/{id}`, `GET /stats`, `GET /metrics`, `POST /threshold`,
`POST /cleanup`, `GET /history`, plus the Update API (`/retrain`, `/retrain/status`).

New (same app, same API key):

| Method | Path | Purpose |
|---|---|---|
| POST | `/detect/url` | Classify one URL; optional `missing_features` and `strategy`; also returns the WAF score of the equivalent request |
| GET | `/url/status` | Models, dataset, features, strategies available |
| GET | `/url/history` | Stored URL analyses |
| POST | `/url/reload` | Reload classifiers after retraining |
| POST | `/experiment/missing-information` | Missing-rate sweep / fixed-feature conditions |
| POST | `/experiment/feature-impact` | Feature-dependency ablation |
| GET | `/experiment/results[/{id}]` | Stored experiments |
| DELETE | `/experiment/results/{id}` | Remove a stored experiment |
| GET | `/`, `/analyze`, `/research`, `/history` | Web UI (overview, analyzer, experiments, persistence) |

Submitted URLs are parsed only — never fetched.

cURL examples (also pre-filled in Swagger at `/docs` → *Try it out*):

```bash
# WAF anomaly model: score an HTTP request (anomalous if score > detection.threshold)
curl -s -X POST localhost:8000/detect -H "X-API-Key: dev-key" -H "Content-Type: application/json" \
  -d '{"method":"GET","path":"/search","query_params":{"q":"<script>alert(1)</script>"},"headers":{},"body":""}'
# → {"is_anomaly":true,"anomaly_score":7.37,...}

curl -s -X POST localhost:8000/detect -H "X-API-Key: dev-key" -H "Content-Type: application/json" \
  -d '{"method":"GET","path":"/products","query_params":{"page":"1","sort":"asc"},"headers":{},"body":""}'
# → {"is_anomaly":false,"anomaly_score":2.22,...}

# Phishing classifier: complete URL
curl -s -X POST localhost:8000/detect/url -H "X-API-Key: dev-key" -H "Content-Type: application/json" \
  -d '{"url":"https://www.readersdigest.co.uk"}'
# → {"is_phishing":false,"phishing_probability":0.001,...}

# Same URL with subdomain + scheme unavailable, baseline model (no handling) → phishing (wrong)
curl -s -X POST localhost:8000/detect/url -H "X-API-Key: dev-key" -H "Content-Type: application/json" \
  -d '{"url":"https://www.readersdigest.co.uk","missing_features":["subdomain","scheme"],"strategy":"none"}'

# Same masked input, robust model (mitigation) → legitimate
curl -s -X POST localhost:8000/detect/url -H "X-API-Key: dev-key" -H "Content-Type: application/json" \
  -d '{"url":"https://www.readersdigest.co.uk","missing_features":["subdomain","scheme"],"strategy":"augmented_training"}'
```

## 10. Tests

```bash
PYTHONPATH=. python -m pytest -q tests
```

Covers URL validation (empty, invalid, very long, query parameters, special
characters), segmentation, missingness (none / one / multiple / all features,
invalid configuration, seeded reproducibility), the tokenizer marker, the causal
WAF objective, metrics, dataset preparation, training + inference of both
variants on a throw-away dataset, experiments + persistence, and every API
endpoint (valid, invalid, unauthenticated). `scripts/run_tests.sh` and
`scripts/eval_benchmark.py` still exercise the running WAF end to end.

## 11. Project layout

```
transformer-waf/
├── src/
│   ├── api/            detection_api.py (WAF + research routes), update_api.py, phishing_routes.py, static/ (UI)
│   ├── preprocessing/  parser, normalizer, tokenizer ([MISSING] marker), compose.py (canonical request text)
│   ├── models/         transformer_model.py (WAFTransformer), url_classifier.py, inference.py, train.py
│   ├── phishing/       url_features, missingness, dataset, training, metrics, experiments, service
│   ├── storage/        detection_store.py (WAF events), experiment_store.py (URL analyses + experiments)
│   └── utils/          config, logging
├── scripts/            prepare_waf_data, train_quick, calibrate_threshold, fetch_phishing_dataset,
│                       train_url_classifier, run_robustness_experiments, demo/start/stop scripts
├── data/               training/ (raw WAF requests), train/ (encoded), phishing/urls.csv (ignored; fetch it)
├── models/             checkpoints/ (WAF), phishing/ (URL classifiers; ignored, train them)
├── tests/
├── integration/        nginx / apache helpers
└── sample_apps/        demo services
```

## 12. Limitations

* **Dataset ceiling.** PhiUSIIL's legitimate URLs are almost all bare
  `https://www.<domain>` home pages (100 % https, 93.7 % `www`, 0 % path in the
  test split), so complete-information accuracy is very high and the
  feature-dependency analysis shows the model leaning on exactly those cues.
  The absolute numbers are specific to this dataset; the degradation/recovery
  *methodology* is not. The pipeline accepts any `url,label` CSV
  (`scripts/fetch_phishing_dataset.py --from-csv`).
* **Segments, not columns.** Because the model is a sequence model, "missing
  information" means missing URL segments. Page-content or WHOIS features that a
  tabular phishing detector would use do not exist here and are not simulated.
* **The WAF anomaly model is unsupervised** and calibrated on a small held-out
  set (331 benign / 599 attack requests from `data/training/`). It flags unusual
  requests; it does not classify phishing.
* **Only the pipeline is tested on the toy data in `tests/`.** All reported
  numbers come from the PhiUSIIL test split.
* Single-process SQLite persistence; experiments run synchronously in a
  worker thread (5 000 URLs × 6 rates × 3 strategies takes a few seconds on CPU).

## 13. Data and model provenance

* PhiUSIIL: Prasad, A. & Chandra, S. (2023). *PhiUSIIL: A diverse security
  profile empowered phishing URL detection framework based on similarity index
  and incremental learning.* Computers & Security. UCI ML Repository,
  https://doi.org/10.24432/C5J92R (CC BY 4.0). Only the URL and label columns are used.
* `models/phishing/*.json` and `experiments.models` (SQLite) record, for each
  checkpoint: variant, training config, seed, epochs, dataset SHA-256,
  vocabulary SHA-256, validation history and training time.
