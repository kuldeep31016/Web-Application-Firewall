"""
End-to-end tests for the research module using a tiny throw-away dataset and
throw-away models trained inside the test session. Nothing here touches the
real dataset or the real checkpoints.
"""

from __future__ import annotations

import os

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.phishing.dataset import DatasetError, assign_splits, load_url_dataset, prepare_url_dataset, dataset_info
from src.phishing.experiments import ExperimentError
from src.phishing.metrics import compute_metrics
from src.phishing.service import PhishingService
from src.phishing.training import TrainConfig, train_url_classifier
from src.storage.experiment_store import ExperimentStore


def _tiny_urls(n_each: int = 120):
    """Structured toy URLs for exercising the pipeline (not a benchmark)."""
    legit = [f"https://www.site{i}.org/page/{i}" for i in range(n_each)]
    phish = [f"http://verify-account{i}.xyz/login.php?id={i}" for i in range(n_each)]
    return legit, phish


@pytest.fixture(scope="module")
def tiny_env(tmp_path_factory):
    base = tmp_path_factory.mktemp("phish")
    legit, phish = _tiny_urls()
    src = base / "source.csv"
    pd.DataFrame({"URL": legit + phish, "Label": ["good"] * len(legit) + ["bad"] * len(phish)}).to_csv(src, index=False)
    ds_path = str(base / "urls.csv")
    info = prepare_url_dataset(str(src), out_path=ds_path, url_col="URL", label_col="Label", positive_value="bad", seed=1)
    models_dir = str(base / "models")
    cfg = TrainConfig(epochs=2, batch_size=32, max_len=32, vocab_size=500, embed_dim=32, num_heads=4, num_layers=1, ff_dim=64, seed=1)
    for variant in ("baseline", "robust"):
        train_url_classifier(variant, cfg, dataset_path=ds_path, models_dir=models_dir)
    store = ExperimentStore(db_path=str(base / "test.db"))
    svc = PhishingService(dataset_path=ds_path, store=store, models_dir=models_dir)
    assert svc.load_models() == {"baseline": True, "robust": True}
    return {"service": svc, "dataset": ds_path, "info": info, "models_dir": models_dir, "store": store}


# ---- dataset -----------------------------------------------------------------

def test_dataset_prepared_with_splits(tiny_env):
    df = load_url_dataset(tiny_env["dataset"])
    assert set(df.columns) == {"url", "label", "split"}
    assert set(df["split"]) == {"train", "val", "test"}
    assert df["url"].is_unique
    info = tiny_env["info"]
    assert info.n_phishing == 120 and info.n_legitimate == 120


def test_split_is_stratified_and_deterministic():
    labels = [0] * 100 + [1] * 50
    a = assign_splits(labels, seed=3)
    b = assign_splits(labels, seed=3)
    assert a == b
    assert a.count("test") == 30 and a.count("val") == 15


def test_missing_dataset_error(tmp_path):
    with pytest.raises(DatasetError):
        load_url_dataset(str(tmp_path / "nope.csv"))


def test_bad_label_value(tmp_path):
    src = tmp_path / "s.csv"
    pd.DataFrame({"URL": ["http://a.com"], "label": ["maybe"]}).to_csv(src, index=False)
    with pytest.raises(DatasetError):
        prepare_url_dataset(str(src), out_path=str(tmp_path / "o.csv"))


# ---- metrics -----------------------------------------------------------------

def test_compute_metrics_values():
    m = compute_metrics([1, 1, 0, 0], [1, 0, 0, 1], [0.9, 0.4, 0.2, 0.6])
    assert m["confusion_matrix"] == {"tp": 1, "fp": 1, "tn": 1, "fn": 1}
    assert m["accuracy"] == 0.5 and m["precision"] == 0.5 and m["recall"] == 0.5 and m["f1"] == 0.5
    assert m["fpr"] == 0.5 and m["fnr"] == 0.5
    assert 0.0 <= m["roc_auc"] <= 1.0


# ---- model inference ---------------------------------------------------------

def test_normal_inference(tiny_env):
    r = tiny_env["service"].analyze_url("https://www.site5.org/page/5", persist=False)
    assert 0.0 <= r["phishing_probability"] <= 1.0
    assert r["model_variant"] == "baseline" and r["missing_features"] == []
    assert r["model_text"] == "https://www.site5.org/page/5"


def test_missing_information_inference(tiny_env):
    r = tiny_env["service"].analyze_url("http://verify-account3.xyz/login.php?id=3", ["scheme", "tld"], "missing_token", persist=False)
    assert r["missing_features"] == ["scheme", "tld"]
    assert "[MISSING]" in r["model_text"] and "xyz" not in r["model_text"] and "http" not in r["model_text"]


def test_mitigated_inference_uses_robust_model(tiny_env):
    r = tiny_env["service"].analyze_url("http://verify-account3.xyz/login.php?id=3", ["domain"], "augmented_training", persist=False)
    assert r["model_variant"] == "robust"


def test_unknown_strategy_and_feature(tiny_env):
    with pytest.raises(ValueError):
        tiny_env["service"].analyze_url("http://a.com", strategy="average", persist=False)
    with pytest.raises(ValueError):
        tiny_env["service"].analyze_url("http://a.com", ["length"], persist=False)


# ---- experiments -------------------------------------------------------------

def test_missing_information_experiment_and_persistence(tiny_env):
    svc = tiny_env["service"]
    exp = svc.run_missing_information(rates=[0.0, 0.5], sample_size=40, seed=3)
    conditions = {(r["condition"], r["strategy"], r["missing_rate"]) for r in exp["results"]}
    assert ("complete", "none", 0.0) in conditions
    assert ("incomplete", "none", 0.5) in conditions
    assert ("mitigated", "augmented_training", 0.5) in conditions
    ref = exp["results"][0]
    assert ref["delta_vs_complete"]["accuracy"] == 0.0
    mitigated = next(r for r in exp["results"] if r["condition"] == "mitigated")
    assert mitigated["recovery_vs_none"] is not None
    # persisted and reloadable
    stored = tiny_env["store"].get_experiment(exp["experiment_id"])
    assert stored is not None and len(stored["results"]) == len(exp["results"])
    assert stored["seed"] == 3 and stored["sample_size"] == 40
    assert stored["results"][-1]["recovery_vs_none"] is not None or stored["results"][-1]["strategy"] != "augmented_training"


def test_experiment_is_reproducible(tiny_env):
    svc = tiny_env["service"]
    a = svc.run_missing_information(rates=[0.3], strategies=["none"], sample_size=40, seed=11, persist=False)
    b = svc.run_missing_information(rates=[0.3], strategies=["none"], sample_size=40, seed=11, persist=False)
    ra = [r for r in a["results"] if r["condition"] == "incomplete"][0]
    rb = [r for r in b["results"] if r["condition"] == "incomplete"][0]
    assert ra["confusion_matrix"] == rb["confusion_matrix"]


def test_feature_impact_experiment(tiny_env):
    exp = tiny_env["service"].run_feature_impact(features=["scheme", "tld"], combinations=[["scheme", "tld"]], sample_size=40, seed=2, persist=False)
    removed = [tuple(r["features"]) for r in exp["results"] if r["condition"] != "complete"]
    assert ("scheme",) in removed and ("tld",) in removed and ("scheme", "tld") in removed
    assert len(exp["summary"]["ranking_by_f1_drop"]) == 3


def test_experiment_validation_errors(tiny_env):
    svc = tiny_env["service"]
    with pytest.raises(ExperimentError):
        svc.run_missing_information(rates=[], persist=False)
    with pytest.raises(ExperimentError):
        svc.run_missing_information(rates=[2.0], persist=False)
    with pytest.raises(ExperimentError):
        svc.run_missing_information(rates=[0.1], strategies=["imputation"], persist=False)
    with pytest.raises(ExperimentError):
        svc.run_missing_information(rates=[0.1], sample_size=1, persist=False)


def test_service_without_models(tmp_path):
    svc = PhishingService(dataset_path=str(tmp_path / "none.csv"), models_dir=str(tmp_path), store=ExperimentStore(db_path=str(tmp_path / "x.db")))
    assert svc.load_models() == {"baseline": False, "robust": False}
    assert svc.status()["ready"] is False
    with pytest.raises(ExperimentError):
        svc.runner()


# ---- API ---------------------------------------------------------------------

@pytest.fixture(scope="module")
def client(tiny_env):
    from src.api import detection_api, phishing_routes

    phishing_routes.configure(tiny_env["service"])  # swap in the tiny service
    with TestClient(detection_api.app) as c:
        yield c
    phishing_routes.configure(detection_api.PHISHING)


H = {"X-API-Key": "dev-key"}


def test_legacy_detect_still_works(client):
    r = client.post("/detect", json={"method": "GET", "path": "/search", "query_params": {"q": "x"}, "headers": {}, "body": ""}, headers=H)
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"is_anomaly", "confidence", "anomaly_score", "timestamp", "request_id"}
    r = client.post("/detect/batch", json=[{"method": "GET", "path": "/", "query_params": {}, "headers": {}, "body": ""}], headers=H)
    assert r.status_code == 200 and len(r.json()) == 1
    assert client.get("/stats", headers=H).status_code == 200
    assert client.get("/logs?limit=5", headers=H).status_code == 200
    assert client.get("/health").json() == {"status": "ok"}


def test_detect_url_endpoint(client):
    r = client.post("/detect/url", json={"url": "https://www.site1.org/page/1"}, headers=H)
    assert r.status_code == 200 and "phishing_probability" in r.json()
    r = client.post("/detect/url", json={"url": "https://www.site1.org/page/1", "missing_features": ["path"], "strategy": "augmented_training"}, headers=H)
    assert r.status_code == 200 and r.json()["model_variant"] == "robust"


def test_detect_url_requires_api_key(client):
    assert client.post("/detect/url", json={"url": "https://a.com"}).status_code == 401


@pytest.mark.parametrize("payload,status", [
    ({"url": ""}, 422),
    ({"url": "not a url"}, 422),
    ({"url": "https://a.com", "missing_features": ["ip"]}, 400),
    ({"url": "https://a.com", "strategy": "mean"}, 400),
    ({}, 422),
])
def test_detect_url_invalid_requests(client, payload, status):
    assert client.post("/detect/url", json=payload, headers=H).status_code == status


def test_experiment_endpoints(client):
    r = client.post("/experiment/missing-information", json={"rates": [0.0, 0.2], "sample_size": 40, "seed": 5}, headers=H)
    assert r.status_code == 200
    exp = r.json()
    assert any(row["condition"] == "mitigated" for row in exp["results"])
    listed = client.get("/experiment/results", headers=H).json()
    assert any(e["experiment_id"] == exp["experiment_id"] for e in listed)
    got = client.get(f"/experiment/results/{exp['experiment_id']}", headers=H).json()
    assert len(got["results"]) == len(exp["results"])
    assert client.get("/experiment/results/does-not-exist", headers=H).status_code == 404
    r = client.post("/experiment/feature-impact", json={"features": ["scheme"], "sample_size": 40}, headers=H)
    assert r.status_code == 200 and len(r.json()["results"]) == 1 + 3
    assert client.delete(f"/experiment/results/{exp['experiment_id']}", headers=H).json() == {"deleted": True}


def test_experiment_invalid_requests(client):
    assert client.post("/experiment/missing-information", json={"rates": []}, headers=H).status_code == 400
    assert client.post("/experiment/missing-information", json={"rates": [0.1], "sample_size": 1}, headers=H).status_code == 422
    assert client.post("/experiment/missing-information", json={"rates": [0.1], "strategies": ["knn"]}, headers=H).status_code == 400
    assert client.post("/experiment/feature-impact", json={"features": ["nope"]}, headers=H).status_code == 400


def test_status_and_history_endpoints(client):
    s = client.get("/url/status", headers=H).json()
    assert s["ready"] is True and [f["name"] for f in s["features"]] == ["scheme", "subdomain", "domain", "tld", "path", "query", "fragment"]
    client.post("/detect/url", json={"url": "https://www.site2.org/page/2"}, headers=H)
    hist = client.get("/url/history", headers=H).json()
    assert hist and hist[0]["url"] == "https://www.site2.org/page/2"


def test_ui_pages_served(client):
    for path in ("/", "/analyze", "/research", "/history"):
        r = client.get(path)
        assert r.status_code == 200 and "Transformer WAF" in r.text
    assert client.get("/static/app.css").status_code == 200
    examples = client.get("/static/examples.json").json()["examples"]
    assert examples and all({"url", "label", "expected", "try"} <= set(e) for e in examples)


def test_detect_url_persist_false_not_stored(client):
    before = len(client.get("/url/history?limit=1000", headers=H).json())
    r = client.post("/detect/url", json={"url": "https://www.site3.org/page/3", "persist": False}, headers=H)
    assert r.status_code == 200
    assert len(client.get("/url/history?limit=1000", headers=H).json()) == before
