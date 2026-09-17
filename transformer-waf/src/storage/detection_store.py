"""
SQLite-backed store for detection events.

Provides simple APIs used by the Detection API for logging, querying,
stats, replay lookup, and cleanup.

Note: the original source of this module was lost in commit 6e85842 (only the
compiled bytecode survived). It has been reconstructed from that bytecode; the
schema and public method signatures are unchanged.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class DetectionStore:
    """SQLite-backed store for detection events.

    Provides simple APIs used by the Detection API for logging, querying,
    stats, replay lookup, and cleanup.
    """

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
                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT UNIQUE,
                    timestamp REAL,
                    method TEXT,
                    path TEXT,
                    anomaly_score REAL,
                    is_anomaly INTEGER,
                    model_version INTEGER,
                    threshold REAL,
                    client_ip_hash TEXT,
                    notes TEXT,
                    created_at TEXT,
                    normalized_request TEXT
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_detections_time ON detections(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_detections_path ON detections(path)")

    def store_detection(
        self,
        *,
        request_id: str,
        method: str,
        path: str,
        query_params: Dict[str, Any],
        headers: Dict[str, Any],
        body: str,
        anomaly_score: float,
        is_anomaly: bool,
        model_version: int,
        threshold: float,
        client_ip: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> None:
        ts = time.time()
        created_at = datetime.now(timezone.utc).isoformat()
        client_ip_hash = hashlib.sha256(client_ip.encode("utf-8")).hexdigest() if client_ip else None
        normalized_request = json.dumps(
            {
                "method": method,
                "path": path,
                "query_params": query_params,
                "headers": headers,
                "body": body or "",
            },
            ensure_ascii=False,
        )
        with self._conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO detections (
                    request_id, timestamp, method, path, anomaly_score, is_anomaly,
                    model_version, threshold, client_ip_hash, notes, created_at, normalized_request
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request_id,
                    ts,
                    method,
                    path,
                    float(anomaly_score),
                    1 if is_anomaly else 0,
                    int(model_version),
                    float(threshold),
                    client_ip_hash,
                    notes,
                    created_at,
                    normalized_request,
                ),
            )

    def get_detections(
        self,
        *,
        from_timestamp: Optional[float] = None,
        to_timestamp: Optional[float] = None,
        is_anomaly: Optional[bool] = None,
        min_score: Optional[float] = None,
        max_score: Optional[float] = None,
        path_pattern: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        where: List[str] = []
        params: List[Any] = []
        if from_timestamp is not None:
            where.append("timestamp >= ?")
            params.append(from_timestamp)
        if to_timestamp is not None:
            where.append("timestamp <= ?")
            params.append(to_timestamp)
        if is_anomaly is not None:
            where.append("is_anomaly = ?")
            params.append(1 if is_anomaly else 0)
        if min_score is not None:
            where.append("anomaly_score >= ?")
            params.append(min_score)
        if max_score is not None:
            where.append("anomaly_score <= ?")
            params.append(max_score)
        if path_pattern:
            where.append("path LIKE ?")
            params.append(f"%{path_pattern}%")
        sql = "SELECT * FROM detections"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([int(limit), int(offset)])
        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    def get_detection_by_id(self, request_id: str) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM detections WHERE request_id = ?", (request_id,)).fetchone()
        return dict(row) if row else None

    def get_stats(self) -> Dict[str, Any]:
        with self._conn() as conn:
            total = conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0] or 0
            anomalies = conn.execute("SELECT COUNT(*) FROM detections WHERE is_anomaly = 1").fetchone()[0] or 0
            avg_score_row = conn.execute("SELECT AVG(anomaly_score) FROM detections").fetchone()
            min_ts_row = conn.execute("SELECT MIN(timestamp) FROM detections").fetchone()
            max_ts_row = conn.execute("SELECT MAX(timestamp) FROM detections").fetchone()
        avg_score = float(avg_score_row[0]) if avg_score_row and avg_score_row[0] is not None else 0.0
        detection_rate = (float(anomalies) / float(total) * 100.0) if total else 0.0
        return {
            "total_requests": int(total),
            "total_anomalies": int(anomalies),
            "avg_score": avg_score,
            "detection_rate": detection_rate,
            "first_detection": min_ts_row[0] if min_ts_row else None,
            "last_detection": max_ts_row[0] if max_ts_row else None,
        }

    def cleanup_old_records(self, retention_days: int = 30) -> int:
        cutoff = time.time() - int(retention_days) * 86400
        with self._conn() as conn:
            cur = conn.execute("DELETE FROM detections WHERE timestamp < ?", (cutoff,))
            return int(cur.rowcount or 0)


detection_store = DetectionStore()

__all__ = ["DetectionStore", "detection_store"]
