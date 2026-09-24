"""
Analytics over the detection database (logs/detections.db).

Every figure is aggregated in SQLite from rows written by the Detection API:

    detections       one row per request scored by the WAF anomaly model
                     (/detect, /detect/batch, /replay)
    url_detections   one row per URL analysed by the phishing classifier
                     (/detect/url with persist=true)

Nothing is estimated or simulated. Live traffic carries no ground-truth labels,
so accuracy-style metrics (precision, recall, FPR) are deliberately absent;
those exist only for the offline experiments on the Research page.

Time handling: timestamps are stored as UTC epoch seconds. Ranges are half-open
[start, end). Buckets are aligned to the caller's local day/hour through a
fixed UTC offset (``tz_offset_min``), so a "day" on the chart is the viewer's
calendar day rather than a UTC day.
"""

from __future__ import annotations

import csv
import io
import math
import sqlite3
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


HOUR = 3600
DAY = 86400

# Where each stored request came from; derived from the `notes` column the API writes.
SOURCE_SQL = (
    "CASE WHEN notes = 'batch_request' THEN 'batch' "
    "WHEN notes LIKE 'replay\\_of\\_%' ESCAPE '\\' THEN 'replay' ELSE 'direct' END"
)

# Older rows stored sha256(ip)[:16], newer rows the full 64-char digest of the same IP.
# Comparing on the 16-char prefix makes both formats count as one client.
CLIENT_SQL = "substr(client_ip_hash, 1, 16)"

# Columns the breakdown table may be grouped by (whitelist; interpolated into SQL).
GROUP_COLUMNS = {
    "path": "path",
    "client": CLIENT_SQL,
    "method": "method",
    "source": SOURCE_SQL,
}

# Sort keys for the breakdown table (whitelist; interpolated into SQL).
SORT_COLUMNS = {
    "key": "k",
    "requests": "n",
    "anomalies": "a",
    "rate": "rate",
    "mean_score": "m",
    "max_score": "mx",
    "change": "(a - COALESCE(pa, 0))",
    "last_seen": "last",
}

# A decision is "near the threshold" when the score is within this fraction of it.
NEAR_THRESHOLD = 0.10


@dataclass
class AnalyticsFilters:
    start: Optional[float] = None      # epoch seconds, inclusive; None = unbounded
    end: Optional[float] = None        # epoch seconds, exclusive; None = unbounded
    tz_offset_min: int = 0             # viewer's offset east of UTC, in minutes
    method: Optional[str] = None
    verdict: Optional[str] = None      # "anomaly" | "normal"
    path: Optional[str] = None         # substring match
    source: Optional[str] = None       # "direct" | "batch" | "replay"

    def previous(self) -> Optional[Tuple[float, float]]:
        """The equal-length period immediately before [start, end), if the range is bounded."""
        if self.start is None or self.end is None:
            return None
        span = self.end - self.start
        return (self.start - span, self.start)


def _like_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _pct_change(cur: Optional[float], prev: Optional[float]) -> Optional[float]:
    """Relative change in percent; None when there is no meaningful base (prev missing or zero)."""
    if cur is None or prev is None or prev == 0:
        return None
    return (cur - prev) / prev * 100.0


def _nice_step(raw: float) -> float:
    if raw <= 0:
        return 1.0
    exp = math.floor(math.log10(raw))
    base = raw / 10 ** exp
    for m in (1, 2, 2.5, 5, 10):
        if base <= m:
            return m * 10 ** exp
    return 10 ** (exp + 1)


class AnalyticsService:
    def __init__(self, db_path: Optional[str] = None) -> None:
        if db_path is None:
            db_path = str(Path(__file__).resolve().parents[2] / "logs" / "detections.db")
        self._db_path = db_path

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _has_table(conn: sqlite3.Connection, name: str) -> bool:
        return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None

    # ------------------------------------------------------------------ SQL helpers

    @staticmethod
    def _where(f: AnalyticsFilters, start: Optional[float], end: Optional[float]) -> Tuple[str, List[Any]]:
        clauses: List[str] = []
        params: List[Any] = []
        if start is not None:
            clauses.append("timestamp >= ?")
            params.append(start)
        if end is not None:
            clauses.append("timestamp < ?")
            params.append(end)
        if f.method:
            clauses.append("method = ?")
            params.append(f.method)
        if f.verdict == "anomaly":
            clauses.append("is_anomaly = 1")
        elif f.verdict == "normal":
            clauses.append("is_anomaly = 0")
        if f.path:
            clauses.append("path LIKE ? ESCAPE '\\'")
            params.append(f"%{_like_escape(f.path)}%")
        if f.source:
            clauses.append(f"({SOURCE_SQL}) = ?")
            params.append(f.source)
        return (" WHERE " + " AND ".join(clauses)) if clauses else "", params

    def _cur_where(self, f: AnalyticsFilters) -> Tuple[str, List[Any]]:
        return self._where(f, f.start, f.end)

    def _prev_where(self, f: AnalyticsFilters) -> Optional[Tuple[str, List[Any]]]:
        prev = f.previous()
        return self._where(f, prev[0], prev[1]) if prev else None

    # ------------------------------------------------------------------ pieces

    def _summary(self, conn: sqlite3.Connection, where: str, params: List[Any]) -> Dict[str, Any]:
        row = conn.execute(
            f"""SELECT COUNT(*) n, COALESCE(SUM(is_anomaly), 0) a, AVG(anomaly_score) m,
                       COUNT(DISTINCT path) paths, COUNT(DISTINCT {CLIENT_SQL}) clients,
                       MIN(timestamp) first, MAX(timestamp) last,
                       MIN(anomaly_score) lo, MAX(anomaly_score) hi
                FROM detections{where}""",
            params,
        ).fetchone()
        n = int(row["n"])
        a = int(row["a"])
        return {
            "requests": n,
            "anomalies": a,
            "normal": n - a,
            "anomaly_rate": (a / n) if n else None,
            "mean_score": row["m"],
            "endpoints": int(row["paths"]),
            "clients": int(row["clients"]),
            "first_seen": row["first"],
            "last_seen": row["last"],
            "min_score": row["lo"],
            "max_score": row["hi"],
        }

    @staticmethod
    def _changes(cur: Dict[str, Any], prev: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if prev is None:
            return {}
        out: Dict[str, Any] = {}
        for key in ("requests", "anomalies", "endpoints", "clients", "mean_score"):
            c, p = cur.get(key), prev.get(key)
            out[key] = {
                "previous": p,
                "abs": (c - p) if (c is not None and p is not None) else None,
                "pct": _pct_change(c, p),
            }
        cr, pr = cur.get("anomaly_rate"), prev.get("anomaly_rate")
        # A rate's change is reported in percentage points, not relative percent.
        out["anomaly_rate"] = {"previous": pr, "pp": (cr - pr) * 100.0 if (cr is not None and pr is not None) else None}
        return out

    @staticmethod
    def _bucket_size(span: float) -> str:
        if span <= 2 * DAY:
            return "hour"
        if span <= 120 * DAY:
            return "day"
        return "week"

    def _timeseries(self, conn: sqlite3.Connection, f: AnalyticsFilters, summary: Dict[str, Any]) -> Dict[str, Any]:
        start = f.start if f.start is not None else summary["first_seen"]
        end = f.end if f.end is not None else (summary["last_seen"] + 1 if summary["last_seen"] is not None else None)
        if start is None or end is None or end <= start:
            return {"bucket": None, "points": []}
        size = self._bucket_size(end - start)
        off = f.tz_offset_min * 60
        # Bucket index for a local-time instant; weeks start on Monday (epoch day 0 was a Thursday).
        if size == "week":
            idx_sql = f"CAST(((timestamp + {off}) / {DAY} + 3) / 7 AS INTEGER)"
            idx_of = lambda t: int(((t + off) // DAY + 3) // 7)  # noqa: E731
            start_of = lambda i: (i * 7 - 3) * DAY - off  # noqa: E731
        else:
            step = HOUR if size == "hour" else DAY
            idx_sql = f"CAST((timestamp + {off}) / {step} AS INTEGER)"
            idx_of = lambda t: int((t + off) // step)  # noqa: E731
            start_of = lambda i: i * step - off  # noqa: E731
        where, params = self._cur_where(f)
        rows = conn.execute(
            f"""SELECT {idx_sql} b, COUNT(*) n, COALESCE(SUM(is_anomaly), 0) a, AVG(anomaly_score) m
                FROM detections{where} GROUP BY b""",
            params,
        ).fetchall()
        by_idx = {int(r["b"]): r for r in rows}
        points = []
        # Emit every bucket in range (zeros included) so gaps in traffic are visible, not interpolated.
        for i in range(idx_of(start), idx_of(end - 1e-6) + 1):
            r = by_idx.get(i)
            n = int(r["n"]) if r else 0
            a = int(r["a"]) if r else 0
            points.append({
                "start": start_of(i),
                "requests": n,
                "anomalies": a,
                "normal": n - a,
                "anomaly_rate": (a / n) if n else None,
                "mean_score": r["m"] if r else None,
            })
        return {"bucket": size, "points": points}

    def _group_counts(self, conn: sqlite3.Connection, expr: str, where: str, params: List[Any]) -> List[Dict[str, Any]]:
        rows = conn.execute(
            f"""SELECT {expr} k, COUNT(*) n, COALESCE(SUM(is_anomaly), 0) a, AVG(anomaly_score) m
                FROM detections{where} GROUP BY k ORDER BY n DESC""",
            params,
        ).fetchall()
        return [
            {"key": r["k"], "requests": int(r["n"]), "anomalies": int(r["a"]),
             "anomaly_rate": int(r["a"]) / int(r["n"]), "mean_score": r["m"]}
            for r in rows
        ]

    def _thresholds(self, conn: sqlite3.Connection, where: str, params: List[Any]) -> List[Dict[str, Any]]:
        rows = conn.execute(
            f"""SELECT threshold t, COUNT(*) n, COALESCE(SUM(is_anomaly), 0) a,
                       MIN(timestamp) first, MAX(timestamp) last
                FROM detections{where} GROUP BY t ORDER BY first""",
            params,
        ).fetchall()
        return [
            {"threshold": r["t"], "requests": int(r["n"]), "anomalies": int(r["a"]),
             "anomaly_rate": int(r["a"]) / int(r["n"]), "first_seen": r["first"], "last_seen": r["last"]}
            for r in rows
        ]

    def _histogram(self, conn: sqlite3.Connection, where: str, params: List[Any], lo: Optional[float], hi: Optional[float]) -> Dict[str, Any]:
        if lo is None or hi is None:
            return {"step": None, "bins": []}
        step = _nice_step((hi - lo) / 12 if hi > lo else 1.0)
        base = math.floor(lo / step) * step
        count = max(1, int(math.floor((hi - base) / step)) + 1)
        rows = conn.execute(
            f"""SELECT MIN(CAST((anomaly_score - ?) / ? AS INTEGER), ?) b,
                       COUNT(*) n, COALESCE(SUM(is_anomaly), 0) a
                FROM detections{where} GROUP BY b""",
            [base, step, count - 1, *params],
        ).fetchall()
        by_idx = {int(r["b"]): r for r in rows}
        bins = []
        for i in range(count):
            r = by_idx.get(i)
            n = int(r["n"]) if r else 0
            a = int(r["a"]) if r else 0
            bins.append({"from": base + i * step, "to": base + (i + 1) * step, "requests": n, "anomalies": a, "normal": n - a})
        return {"step": step, "bins": bins}

    @staticmethod
    def _quantile(conn: sqlite3.Connection, where: str, params: List[Any], n: int, q: float) -> Optional[float]:
        """Linear-interpolated quantile computed with two ordered lookups (no full fetch)."""
        if n == 0:
            return None
        pos = q * (n - 1)
        lo_i, hi_i = int(math.floor(pos)), int(math.ceil(pos))
        sql = f"SELECT anomaly_score FROM detections{where} ORDER BY anomaly_score LIMIT 1 OFFSET ?"
        lo_v = conn.execute(sql, [*params, lo_i]).fetchone()[0]
        if hi_i == lo_i:
            return lo_v
        hi_v = conn.execute(sql, [*params, hi_i]).fetchone()[0]
        return lo_v + (hi_v - lo_v) * (pos - lo_i)

    def _score_stats(self, conn: sqlite3.Connection, f: AnalyticsFilters) -> List[Dict[str, Any]]:
        out = []
        for label, verdict in (("all", None), ("anomaly", "anomaly"), ("normal", "normal")):
            if f.verdict and verdict and f.verdict != verdict:
                continue
            where, params = self._cur_where(replace(f, verdict=verdict or f.verdict))
            row = conn.execute(
                f"""SELECT COUNT(*) n, MIN(anomaly_score) lo, MAX(anomaly_score) hi, AVG(anomaly_score) m,
                           AVG(anomaly_score - threshold) margin
                    FROM detections{where}""",
                params,
            ).fetchone()
            n = int(row["n"])
            out.append({
                "group": label, "requests": n, "min": row["lo"], "max": row["hi"], "mean": row["m"],
                "median": self._quantile(conn, where, params, n, 0.5),
                "p95": self._quantile(conn, where, params, n, 0.95),
                "mean_margin": row["margin"],
            })
        return out

    def _near_threshold(self, conn: sqlite3.Connection, where: str, params: List[Any]) -> Dict[str, int]:
        where2 = where + (" AND " if where else " WHERE ") + "threshold > 0"
        row = conn.execute(
            f"""SELECT COALESCE(SUM(CASE WHEN is_anomaly = 1 AND anomaly_score < threshold * (1 + ?) THEN 1 ELSE 0 END), 0) flagged,
                       COALESCE(SUM(CASE WHEN is_anomaly = 0 AND anomaly_score >= threshold * (1 - ?) THEN 1 ELSE 0 END), 0) passed
                FROM detections{where2}""",
            [NEAR_THRESHOLD, NEAR_THRESHOLD, *params],
        ).fetchone()
        return {"band": NEAR_THRESHOLD, "flagged": int(row["flagged"]), "passed": int(row["passed"])}

    def _new_attacked_paths(self, conn: sqlite3.Connection, f: AnalyticsFilters) -> Optional[List[str]]:
        prev = self._prev_where(f)
        if prev is None:
            return None
        cw, cp = self._cur_where(f)
        pw, pp = prev
        rows = conn.execute(
            f"""SELECT path FROM detections{cw} {'AND' if cw else 'WHERE'} is_anomaly = 1
                EXCEPT SELECT path FROM detections{pw} {'AND' if pw else 'WHERE'} is_anomaly = 1""",
            [*cp, *pp],
        ).fetchall()
        return sorted(r[0] for r in rows)

    def _url_analyses(self, conn: sqlite3.Connection, f: AnalyticsFilters) -> Optional[Dict[str, Any]]:
        """URL-classifier activity. Only the date range applies; request filters do not map onto URLs."""
        if not self._has_table(conn, "url_detections"):
            return None

        def where_for(start: Optional[float], end: Optional[float]) -> Tuple[str, List[Any]]:
            c, p = [], []
            if start is not None:
                c.append("timestamp >= ?"); p.append(start)
            if end is not None:
                c.append("timestamp < ?"); p.append(end)
            return (" WHERE " + " AND ".join(c)) if c else "", p

        def totals(where: str, params: List[Any]) -> Dict[str, Any]:
            r = conn.execute(
                f"""SELECT COUNT(*) n, COALESCE(SUM(is_phishing), 0) ph, AVG(phishing_probability) prob,
                           AVG(inference_ms) ms,
                           COALESCE(SUM(CASE WHEN missing_features IS NOT NULL AND missing_features != '[]' THEN 1 ELSE 0 END), 0) incomplete,
                           COALESCE(SUM(CASE WHEN waf_anomaly_score IS NOT NULL THEN 1 ELSE 0 END), 0) waf_scored,
                           COALESCE(SUM(CASE WHEN waf_is_anomaly = 1 THEN 1 ELSE 0 END), 0) waf_flagged,
                           COALESCE(SUM(CASE WHEN waf_is_anomaly = 1 AND is_phishing = 1 THEN 1 ELSE 0 END), 0) both_flagged
                    FROM url_detections{where}""",
                params,
            ).fetchone()
            n = int(r["n"])
            return {
                "analyses": n, "phishing": int(r["ph"]), "phishing_rate": int(r["ph"]) / n if n else None,
                "mean_probability": r["prob"], "mean_inference_ms": r["ms"],
                "incomplete": int(r["incomplete"]), "waf_scored": int(r["waf_scored"]),
                "waf_flagged": int(r["waf_flagged"]), "both_flagged": int(r["both_flagged"]),
            }

        cw, cp = where_for(f.start, f.end)
        cur = totals(cw, cp)
        prev_range = f.previous()
        prev = totals(*where_for(*prev_range)) if prev_range else None
        strategies = [
            {"strategy": r["s"], "model_variant": r["v"], "analyses": int(r["n"]), "phishing": int(r["ph"]),
             "phishing_rate": int(r["ph"]) / int(r["n"]), "mean_probability": r["prob"], "mean_inference_ms": r["ms"]}
            for r in conn.execute(
                f"""SELECT strategy s, model_variant v, COUNT(*) n, COALESCE(SUM(is_phishing), 0) ph,
                           AVG(phishing_probability) prob, AVG(inference_ms) ms
                    FROM url_detections{cw} GROUP BY s, v ORDER BY n DESC""",
                cp,
            ).fetchall()
        ]
        missing = [
            {"feature": r["feature"], "analyses": int(r["n"])}
            for r in conn.execute(
                f"""SELECT j.value feature, COUNT(*) n
                    FROM url_detections u, json_each(CASE WHEN json_valid(u.missing_features) THEN u.missing_features ELSE '[]' END) j
                    {cw} GROUP BY j.value ORDER BY n DESC""",
                cp,
            ).fetchall()
        ]
        return {
            "summary": cur,
            "previous": prev,
            "change": {"analyses": _pct_change(cur["analyses"], prev["analyses"]) if prev else None},
            "strategies": strategies,
            "missing_features": missing,
        }

    # ------------------------------------------------------------------ public API

    def filter_options(self, conn: sqlite3.Connection) -> Dict[str, List[str]]:
        methods = [r[0] for r in conn.execute("SELECT DISTINCT method FROM detections ORDER BY method").fetchall()]
        sources = [r[0] for r in conn.execute(f"SELECT DISTINCT {SOURCE_SQL} FROM detections").fetchall()]
        order = {"direct": 0, "batch": 1, "replay": 2}
        return {"methods": methods, "sources": sorted(sources, key=lambda s: order.get(s, 9))}

    def overview(self, f: AnalyticsFilters) -> Dict[str, Any]:
        with self._conn() as conn:
            cw, cp = self._cur_where(f)
            summary = self._summary(conn, cw, cp)
            prev_w = self._prev_where(f)
            prev_summary = self._summary(conn, *prev_w) if prev_w else None
            timeseries = self._timeseries(conn, f, summary)
            methods = self._group_counts(conn, "method", cw, cp)
            sources = self._group_counts(conn, SOURCE_SQL, cw, cp)
            thresholds = self._thresholds(conn, cw, cp)
            histogram = self._histogram(conn, cw, cp, summary["min_score"], summary["max_score"])
            score_stats = self._score_stats(conn, f)
            near = self._near_threshold(conn, cw, cp)
            top = self._breakdown(conn, f, "path", "anomalies", "desc", None, 8, 0)
            clients = self._breakdown(conn, f, "client", "anomalies", "desc", None, 3, 0)
            new_paths = self._new_attacked_paths(conn, f)
            prev_thresholds = self._thresholds(conn, *prev_w) if prev_w else []
            urls = self._url_analyses(conn, f)
            options = self.filter_options(conn)
        prev = f.previous()
        result = {
            "range": {"start": f.start, "end": f.end, "tz_offset_min": f.tz_offset_min},
            "previous_range": {"start": prev[0], "end": prev[1]} if prev else None,
            "summary": summary,
            "previous": prev_summary,
            "change": self._changes(summary, prev_summary),
            "timeseries": timeseries,
            "distributions": {"methods": methods, "sources": sources, "thresholds": thresholds, "score_histogram": histogram},
            "score_stats": score_stats,
            "near_threshold": near,
            "top_endpoints": top["rows"],
            "urls": urls,
            "filter_options": options,
        }
        result["insights"] = self._insights(result, clients["rows"], new_paths, prev_thresholds)
        return result

    def _breakdown(self, conn: sqlite3.Connection, f: AnalyticsFilters, group_by: str, sort: str, order: str,
                   search: Optional[str], limit: Optional[int], offset: int) -> Dict[str, Any]:
        key = GROUP_COLUMNS[group_by]
        sort_sql = SORT_COLUMNS[sort]
        direction = "ASC" if order == "asc" else "DESC"
        cw, cp = self._cur_where(f)
        prev_w = self._prev_where(f)
        pw, pp = prev_w if prev_w else (" WHERE 0", [])
        search_sql, sp = "", []
        if search:
            search_sql = " WHERE COALESCE(k, '') LIKE ? ESCAPE '\\'"
            sp = [f"%{_like_escape(search)}%"]
        base = f"""
            WITH cur AS (SELECT {key} k, COUNT(*) n, COALESCE(SUM(is_anomaly), 0) a, AVG(anomaly_score) m,
                                MAX(anomaly_score) mx, MAX(timestamp) last
                         FROM detections{cw} GROUP BY k),
                 prev AS (SELECT {key} k, COUNT(*) n, COALESCE(SUM(is_anomaly), 0) a FROM detections{pw} GROUP BY k),
                 j AS (SELECT cur.*, prev.n pn, prev.a pa, CAST(cur.a AS REAL) / cur.n rate
                       FROM cur LEFT JOIN prev ON cur.k IS prev.k)
        """
        total = conn.execute(base + f"SELECT COUNT(*) FROM j{search_sql}", [*cp, *pp, *sp]).fetchone()[0]
        page = f" LIMIT {int(limit)} OFFSET {int(offset)}" if limit is not None else ""
        rows = conn.execute(
            base + f"SELECT * FROM j{search_sql} ORDER BY {sort_sql} {direction}, n DESC, k ASC{page}",
            [*cp, *pp, *sp],
        ).fetchall()
        has_prev = prev_w is not None
        return {
            "group_by": group_by,
            "total": int(total),
            "has_previous": has_prev,
            "rows": [
                {
                    "key": r["k"],
                    "requests": int(r["n"]),
                    "anomalies": int(r["a"]),
                    "anomaly_rate": r["rate"],
                    "mean_score": r["m"],
                    "max_score": r["mx"],
                    "last_seen": r["last"],
                    "previous_requests": int(r["pn"] or 0) if has_prev else None,
                    "previous_anomalies": int(r["pa"] or 0) if has_prev else None,
                    "anomalies_change": (int(r["a"]) - int(r["pa"] or 0)) if has_prev else None,
                }
                for r in rows
            ],
        }

    def breakdown(self, f: AnalyticsFilters, group_by: str = "path", sort: str = "anomalies", order: str = "desc",
                  search: Optional[str] = None, limit: Optional[int] = 25, offset: int = 0) -> Dict[str, Any]:
        if group_by not in GROUP_COLUMNS:
            raise ValueError(f"group_by must be one of {sorted(GROUP_COLUMNS)}")
        if sort not in SORT_COLUMNS:
            raise ValueError(f"sort must be one of {sorted(SORT_COLUMNS)}")
        with self._conn() as conn:
            return self._breakdown(conn, f, group_by, sort, order, search, limit, offset)

    def breakdown_csv(self, f: AnalyticsFilters, group_by: str = "path", sort: str = "anomalies", order: str = "desc",
                      search: Optional[str] = None) -> str:
        data = self.breakdown(f, group_by, sort, order, search, limit=None)
        buf = io.StringIO()
        w = csv.writer(buf)
        header = [group_by, "requests", "anomalies", "anomaly_rate_pct", "mean_score", "max_score", "last_seen_utc"]
        if data["has_previous"]:
            header += ["previous_requests", "previous_anomalies", "anomalies_change"]
        w.writerow(header)
        for r in data["rows"]:
            row = [
                r["key"] if r["key"] is not None else "",
                r["requests"], r["anomalies"],
                f"{r['anomaly_rate'] * 100:.2f}" if r["anomaly_rate"] is not None else "",
                f"{r['mean_score']:.4f}" if r["mean_score"] is not None else "",
                f"{r['max_score']:.4f}" if r["max_score"] is not None else "",
                datetime.fromtimestamp(r["last_seen"], tz=timezone.utc).isoformat() if r["last_seen"] is not None else "",
            ]
            if data["has_previous"]:
                row += [r["previous_requests"], r["previous_anomalies"], r["anomalies_change"]]
            w.writerow(row)
        return buf.getvalue()

    # ------------------------------------------------------------------ insights

    @staticmethod
    def _insights(d: Dict[str, Any], top_clients: List[Dict[str, Any]], new_paths: Optional[List[str]],
                  prev_thresholds: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Deterministic observations; each rule fires only when the numbers support it."""
        out: List[Dict[str, str]] = []
        s, prev, ch = d["summary"], d["previous"], d["change"]
        n, a = s["requests"], s["anomalies"]

        def add(level: str, text: str) -> None:
            out.append({"level": level, "text": text})

        if n == 0:
            return out

        # Volume vs the previous period.
        if prev is not None:
            pn = prev["requests"]
            if pn == 0:
                add("info", f"No requests were scored in the previous period; all {n:,} in this range are new activity.")
            else:
                pct = ch["requests"]["pct"]
                if pct is not None and abs(pct) >= 10:
                    add("info", f"Scored traffic {'rose' if pct > 0 else 'fell'} {abs(pct):.0f}% versus the previous period ({pn:,} → {n:,}).")
            pp = ch["anomaly_rate"]["pp"]
            if pp is not None and abs(pp) >= 5 and prev["requests"] >= 5:
                cur_t = {t["threshold"] for t in d["distributions"]["thresholds"]}
                prev_t = {t["threshold"] for t in prev_thresholds}
                caveat = (" The two periods were scored with different thresholds, so part of this shift may come from "
                          "the threshold rather than the traffic.") if cur_t != prev_t else ""
                add("warn" if pp > 0 else "good",
                    f"The anomaly rate {'increased' if pp > 0 else 'decreased'} by {abs(pp):.1f} percentage points "
                    f"({prev['anomaly_rate'] * 100:.1f}% → {s['anomaly_rate'] * 100:.1f}%).{caveat}")

        # Where the anomalies concentrate.
        top = d["top_endpoints"]
        if a > 0 and top and top[0]["anomalies"] > 0:
            share = top[0]["anomalies"] / a
            if share >= 0.4 and a >= 5 and top[0]["anomalies"] < a:
                add("warn", f"{top[0]['key']} accounts for {top[0]['anomalies']:,} of {a:,} anomalies ({share * 100:.0f}%) — "
                            "the most targeted endpoint in this range.")
            risky = [t for t in top if t["requests"] >= 5]
            if risky:
                worst = max(risky, key=lambda t: (t["anomaly_rate"], t["requests"]))
                if worst["anomaly_rate"] >= 0.5:
                    add("warn", f"{worst['anomaly_rate'] * 100:.0f}% of requests to {worst['key']} were flagged "
                                f"({worst['anomalies']:,} of {worst['requests']:,}).")

        # One client driving most of the flagged traffic.
        if a >= 5 and s["clients"] > 1 and top_clients and top_clients[0]["key"] and top_clients[0]["anomalies"] / a >= 0.5:
            c = top_clients[0]
            add("warn", f"A single client (hashed IP {c['key'][:10]}…) sent {c['anomalies']:,} of {a:,} flagged requests "
                        f"({c['anomalies'] / a * 100:.0f}%).")

        if new_paths and prev is not None and prev["requests"] > 0:
            shown = ", ".join(new_paths[:3]) + (f" and {len(new_paths) - 3} more" if len(new_paths) > 3 else "")
            add("info", f"{len(new_paths)} endpoint{'s' if len(new_paths) != 1 else ''} received anomalous requests "
                        f"that had none in the previous period: {shown}.")

        # Busiest bucket for anomalies.
        pts = [p for p in d["timeseries"]["points"] if p["anomalies"] > 0]
        if len(pts) >= 2:
            peak = max(pts, key=lambda p: p["anomalies"])
            if peak["anomalies"] / a >= 0.3:
                bucket = d["timeseries"]["bucket"]
                local = datetime.fromtimestamp(peak["start"], tz=timezone(timedelta(minutes=d["range"]["tz_offset_min"])))
                when = local.strftime("%d %b %Y %H:00" if bucket == "hour" else "%d %b %Y")
                label = {"hour": "the hour starting", "day": "a single day,", "week": "the week of"}[bucket]
                add("info", f"Anomalies peaked in {label} {when}: {peak['anomalies']:,} of {a:,} "
                            f"({peak['anomalies'] / a * 100:.0f}%) fall in that {bucket}.")

        # Decisions close to the threshold.
        near = d["near_threshold"]
        close = near["flagged"] + near["passed"]
        if close > 0 and close / n >= 0.05:
            add("info", f"{close:,} decisions ({close / n * 100:.0f}%) scored within {near['band'] * 100:.0f}% of their threshold "
                        f"({near['flagged']:,} flagged, {near['passed']:,} passed) — these are the most sensitive to threshold changes.")

        # Score separation between the two verdicts.
        stats = {g["group"]: g for g in d["score_stats"]}
        an, no = stats.get("anomaly"), stats.get("normal")
        if an and no and an["requests"] and no["requests"] and an["median"] is not None and no["median"] is not None:
            add("info", f"Median anomaly score: {an['median']:.2f} for flagged requests vs {no['median']:.2f} for normal ones.")

        # Mixed thresholds make rates across time not like-for-like.
        th = d["distributions"]["thresholds"]
        if len(th) > 1:
            lo, hi = min(t["threshold"] for t in th), max(t["threshold"] for t in th)
            add("warn", f"Requests in this range were scored under {len(th)} different thresholds ({lo:g}–{hi:g}); "
                        "anomaly rates from different thresholds are not directly comparable.")

        if n < 20:
            add("info", f"Small sample: only {n:,} request{'s' if n != 1 else ''} in this range, so percentages can swing sharply.")
        return out
