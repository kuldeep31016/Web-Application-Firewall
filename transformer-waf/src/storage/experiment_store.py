"""
SQLite persistence for the phishing research module.

Lives in the same database file as ``DetectionStore`` (logs/detections.db) and
follows the same conventions. Three tables:

    url_detections      one row per single-URL analysis (history for the UI)
    experiments         one row per experiment run (configuration + provenance)
    experiment_results  one row per evaluated condition inside an experiment
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


class ExperimentStore:
    def __init__(self, db_path: Optional[str] = None) -> None:
        if db_path is None:
            base_dir = Path(__file__).resolve().parents[2]
            logs_dir = base_dir / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(logs_dir / "detections.db")
        self._db_path = db_path
        self._ensure_schema()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, isolation_level=None)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS url_detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT UNIQUE,
                    timestamp REAL,
                    url TEXT,
                    model_text TEXT,
                    missing_features TEXT,
                    strategy TEXT,
                    model_variant TEXT,
                    phishing_probability REAL,
                    is_phishing INTEGER,
                    threshold REAL,
                    waf_anomaly_score REAL,
                    waf_is_anomaly INTEGER,
                    inference_ms REAL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_url_detections_time ON url_detections(timestamp)")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT UNIQUE,
                    kind TEXT,
                    created_at REAL,
                    seed INTEGER,
                    dataset_path TEXT,
                    dataset_sha256 TEXT,
                    evaluation_split TEXT,
                    sample_size INTEGER,
                    config TEXT,
                    models TEXT,
                    summary TEXT,
                    duration_seconds REAL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS experiment_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT,
                    condition TEXT,
                    strategy TEXT,
                    model_variant TEXT,
                    representation TEXT,
                    missing_rate REAL,
                    features TEXT,
                    realised_missing_rate REAL,
                    n INTEGER,
                    accuracy REAL,
                    precision REAL,
                    recall REAL,
                    f1 REAL,
                    fpr REAL,
                    fnr REAL,
                    roc_auc REAL,
                    tp INTEGER, fp INTEGER, tn INTEGER, fn INTEGER,
                    delta_vs_complete TEXT,
                    extra TEXT
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_experiment_results_exp ON experiment_results(experiment_id)")

    # ----- URL detections -------------------------------------------------

    def store_url_detection(self, record: Dict[str, Any]) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO url_detections (
                    request_id, timestamp, url, model_text, missing_features, strategy, model_variant,
                    phishing_probability, is_phishing, threshold, waf_anomaly_score, waf_is_anomaly, inference_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record["request_id"],
                    float(record.get("timestamp") or time.time()),
                    record["url"],
                    record.get("model_text"),
                    json.dumps(sorted(record.get("missing_features") or [])),
                    record.get("strategy"),
                    record.get("model_variant"),
                    float(record["phishing_probability"]),
                    1 if record["is_phishing"] else 0,
                    float(record.get("threshold", 0.5)),
                    record.get("waf_anomaly_score"),
                    None if record.get("waf_is_anomaly") is None else (1 if record["waf_is_anomaly"] else 0),
                    record.get("inference_ms"),
                ),
            )

    def get_url_detections(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM url_detections ORDER BY timestamp DESC LIMIT ? OFFSET ?", (int(limit), int(offset))
            ).fetchall()
        out = []
        for row in rows:
            d = dict(row)
            d["missing_features"] = json.loads(d.get("missing_features") or "[]")
            d["is_phishing"] = bool(d["is_phishing"])
            d["waf_is_anomaly"] = None if d["waf_is_anomaly"] is None else bool(d["waf_is_anomaly"])
            out.append(d)
        return out

    def url_detection_stats(self) -> Dict[str, Any]:
        with self._conn() as conn:
            total = conn.execute("SELECT COUNT(*) FROM url_detections").fetchone()[0] or 0
            phishing = conn.execute("SELECT COUNT(*) FROM url_detections WHERE is_phishing = 1").fetchone()[0] or 0
            avg_ms = conn.execute("SELECT AVG(inference_ms) FROM url_detections").fetchone()[0]
            last = conn.execute("SELECT MAX(timestamp) FROM url_detections").fetchone()[0]
        return {
            "total_urls": int(total),
            "phishing_detected": int(phishing),
            "avg_inference_ms": float(avg_ms) if avg_ms is not None else None,
            "last_analysis": last,
        }

    # ----- Experiments ----------------------------------------------------

    def store_experiment(self, experiment: Dict[str, Any], results: List[Dict[str, Any]]) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO experiments (
                    experiment_id, kind, created_at, seed, dataset_path, dataset_sha256, evaluation_split,
                    sample_size, config, models, summary, duration_seconds
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    experiment["experiment_id"],
                    experiment["kind"],
                    float(experiment.get("created_at") or time.time()),
                    int(experiment.get("seed", 0)),
                    experiment.get("dataset_path"),
                    experiment.get("dataset_sha256"),
                    experiment.get("evaluation_split", "test"),
                    int(experiment.get("sample_size") or 0),
                    json.dumps(experiment.get("config") or {}),
                    json.dumps(experiment.get("models") or {}),
                    json.dumps(experiment.get("summary") or {}),
                    experiment.get("duration_seconds"),
                ),
            )
            conn.execute("DELETE FROM experiment_results WHERE experiment_id = ?", (experiment["experiment_id"],))
            for r in results:
                cm = r.get("confusion_matrix") or {}
                extra = dict(r.get("extra") or {})
                if r.get("recovery_vs_none") is not None:
                    extra["recovery_vs_none"] = r["recovery_vs_none"]
                conn.execute(
                    """
                    INSERT INTO experiment_results (
                        experiment_id, condition, strategy, model_variant, representation, missing_rate, features,
                        realised_missing_rate, n, accuracy, precision, recall, f1, fpr, fnr, roc_auc,
                        tp, fp, tn, fn, delta_vs_complete, extra
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        experiment["experiment_id"],
                        r.get("condition"),
                        r.get("strategy"),
                        r.get("model_variant"),
                        r.get("representation"),
                        float(r.get("missing_rate") or 0.0),
                        json.dumps(r.get("features") or []),
                        r.get("realised_missing_rate"),
                        int(r.get("n") or 0),
                        r.get("accuracy"),
                        r.get("precision"),
                        r.get("recall"),
                        r.get("f1"),
                        r.get("fpr"),
                        r.get("fnr"),
                        r.get("roc_auc"),
                        cm.get("tp"),
                        cm.get("fp"),
                        cm.get("tn"),
                        cm.get("fn"),
                        json.dumps(r.get("delta_vs_complete") or {}),
                        json.dumps(extra),
                    ),
                )

    @staticmethod
    def _row_to_result(row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        extra = json.loads(d.get("extra") or "{}")
        recovery = extra.pop("recovery_vs_none", None)
        return {
            "condition": d["condition"],
            "strategy": d["strategy"],
            "model_variant": d["model_variant"],
            "representation": d["representation"],
            "missing_rate": d["missing_rate"],
            "features": json.loads(d.get("features") or "[]"),
            "realised_missing_rate": d["realised_missing_rate"],
            "n": d["n"],
            "accuracy": d["accuracy"],
            "precision": d["precision"],
            "recall": d["recall"],
            "f1": d["f1"],
            "fpr": d["fpr"],
            "fnr": d["fnr"],
            "roc_auc": d["roc_auc"],
            "confusion_matrix": {"tp": d["tp"], "fp": d["fp"], "tn": d["tn"], "fn": d["fn"]},
            "delta_vs_complete": json.loads(d.get("delta_vs_complete") or "{}"),
            "recovery_vs_none": recovery,
            "extra": extra,
        }

    @staticmethod
    def _row_to_experiment(row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        return {
            "experiment_id": d["experiment_id"],
            "kind": d["kind"],
            "created_at": d["created_at"],
            "seed": d["seed"],
            "dataset_path": d["dataset_path"],
            "dataset_sha256": d["dataset_sha256"],
            "evaluation_split": d["evaluation_split"],
            "sample_size": d["sample_size"],
            "config": json.loads(d.get("config") or "{}"),
            "models": json.loads(d.get("models") or "{}"),
            "summary": json.loads(d.get("summary") or "{}"),
            "duration_seconds": d["duration_seconds"],
        }

    def list_experiments(self, limit: int = 50, kind: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            if kind:
                rows = conn.execute(
                    "SELECT * FROM experiments WHERE kind = ? ORDER BY created_at DESC LIMIT ?", (kind, int(limit))
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM experiments ORDER BY created_at DESC LIMIT ?", (int(limit),)).fetchall()
        return [self._row_to_experiment(r) for r in rows]

    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM experiments WHERE experiment_id = ?", (experiment_id,)).fetchone()
            if not row:
                return None
            results = conn.execute(
                "SELECT * FROM experiment_results WHERE experiment_id = ? ORDER BY id", (experiment_id,)
            ).fetchall()
        exp = self._row_to_experiment(row)
        exp["results"] = [self._row_to_result(r) for r in results]
        return exp

    def delete_experiment(self, experiment_id: str) -> bool:
        with self._conn() as conn:
            cur = conn.execute("DELETE FROM experiments WHERE experiment_id = ?", (experiment_id,))
            conn.execute("DELETE FROM experiment_results WHERE experiment_id = ?", (experiment_id,))
            return bool(cur.rowcount)


experiment_store = ExperimentStore()

__all__ = ["ExperimentStore", "experiment_store"]
