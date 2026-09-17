---
title: "Verified Example URLs for the Analyzer"
subtitle: "What to type, what to tick, what the model will say"
date: "September 2026"
---

All results below were produced by the shipped checkpoints on 17 Sept 2026
(baseline trained 2026-09-16 21:46, robust 2026-09-16 22:02). The same list is
available in the **Load an example** dropdown on the Analyze page, which also
pre-ticks the suggested "unavailable" segments.

"Ground truth" is the label in the PhiUSIIL test split (never used for training)
or *hand-written* for URLs that are not in the dataset. Strategy names are the
radio buttons on the Analyze page:

* **None** = "None (blank the segment)" — baseline model
* **Marker** = "Explicit [MISSING] marker" — baseline model
* **Mitigation** = "Training with incomplete examples" — robust model

# Legitimate URLs

| URL | Ground truth | Complete | Tick *subdomain + scheme* → None | → Mitigation |
|---|---|---|---|---|
| `https://www.wikipedia.org` | hand-written | Legitimate 0.1 % | **Phishing 99 %** (wrong) | Legitimate 2 % |
| `https://www.readersdigest.co.uk` | legitimate | Legitimate 0.1 % | **Phishing 100 %** (wrong) | Legitimate 7 % |
| `https://www.lanazione.it` | legitimate | Legitimate 0.1 % | **Phishing 100 %** (wrong) | Legitimate 3 % |
| `https://www.csas.ed.ac.uk` | legitimate | Legitimate 0.1 % | Legitimate 27 % | Legitimate 1 % |

These show the research effect in one click: remove `www` and `https` and the
baseline model collapses; the robust model does not.

# Phishing URLs

| URL | Ground truth | Complete | Tick *path + scheme* → None | → Mitigation |
|---|---|---|---|---|
| `https://veriify-confirm-info.start.page/` | phishing | Phishing 100 % | Phishing 100 % | Phishing 99.9 % |
| `http://www.altiusinstitute.in/ss/delvqe&&wpums8/bellzall.php?email=` | phishing | Phishing 100 % | Phishing 98 % | Phishing 99.9 % |
| `https://wallets-2mz.pages.dev/` | phishing | Phishing 100 % | Phishing 99 % | Phishing 100 % |
| `http://www.kovacekr.ga` | phishing | Phishing 100 % | **Legitimate 0.1 %** (missed) | Phishing 95 % |

The last row is the mirror image of the legitimate cases: with path and scheme
unavailable the baseline model misses the phishing URL; the robust model catches it.

# Hand-written phishing-style URLs

| URL | Complete | Any segments unavailable |
|---|---|---|
| `http://secure-login.paypa1-verify.tk/account/update.php?id=99` | Phishing 100 % | Phishing ≥ 99 % with domain+scheme, tld, path+query or subdomain+scheme+path unavailable (both models); the WAF also flags the equivalent request |
| `http://appleid-apple.com-secure-check.xyz/login` | Phishing 100 % | Phishing ≥ 97 % under the same masks |
| `http://192.168.10.5/admin/verify.php` | Phishing 100 % | Phishing ≥ 95 % under the same masks |

# Known limitation — show it deliberately

| URL | Result | Why |
|---|---|---|
| `https://www.github.com/login` | **Phishing 100 %** (false positive) | No legitimate URL in the PhiUSIIL training data has a path; the model learned "path ⇒ phishing". Same for `https://www.amazon.in/gp/help/customer/display.html`. |

This is a property of the dataset, documented in `README.md` §12. The training
pipeline accepts any other `url,label` CSV via
`scripts/fetch_phishing_dataset.py --from-csv`, which is how this would be fixed
in a follow-up.

# WAF request examples (bottom of the Analyze page)

| Method / path | Query (one per line) | Expected |
|---|---|---|
| `GET /search` | `q=<script>alert(1)</script>` | Anomalous (score ≈ 7.4, threshold 3.06) |
| `GET /login` | `id=' OR '1'='1` | Anomalous (≈ 6.4) |
| `GET /download` | `file=../../../../etc/passwd` | Anomalous (≈ 5.5) |
| `GET /search` | `q=hello world` | Normal (≈ 2.6) |
| `GET /products` | `page=1` and `sort=asc` | Normal (≈ 2.2) |

The anomaly score is the mean next-token negative log-likelihood under a model
trained only on benign requests — "how unusual is this request", not "is this
phishing".

# Invalid inputs (the UI must show an error, not crash)

| Input | Expected message |
|---|---|
| empty box | "Enter a URL to analyze." |
| `not a url` (has a space) | 422 — "URL contains whitespace or control characters" |
| `http://` | 422 — "URL has no host" |
| 3,000-character URL | 422 — "URL is longer than 2048 characters" |
| wrong API key | 401 — "Unauthorized" |
