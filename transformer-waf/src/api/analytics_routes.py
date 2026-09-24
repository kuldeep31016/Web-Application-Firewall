"""
Analytics routes, mounted on the existing WAF Detection API (same port, same
X-API-Key header). All figures are aggregated from logs/detections.db by
``src.analytics.service.AnalyticsService``; nothing here is simulated.
"""

from __future__ import annotations

from typing import Dict, Literal, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from ..analytics.service import AnalyticsFilters, AnalyticsService


# Authentication is applied by the detection API when it includes this router
# (``dependencies=[Security(verify_api_key)]``), exactly like the research routes.
router = APIRouter(prefix="/analytics", tags=["analytics"])
_service: Optional[AnalyticsService] = None

GroupBy = Literal["path", "client", "method", "source"]
SortKey = Literal["key", "requests", "anomalies", "rate", "mean_score", "max_score", "change", "last_seen"]


def configure(service: AnalyticsService) -> None:
    global _service
    _service = service


def _svc() -> AnalyticsService:
    if _service is None:
        raise HTTPException(status_code=503, detail="Analytics service not configured")
    return _service


def _filters(start: Optional[float], end: Optional[float], tz_offset: int, method: Optional[str],
             verdict: Optional[str], path: Optional[str], source: Optional[str]) -> AnalyticsFilters:
    """start/end are epoch seconds ([start, end)); both omitted means all time. tz_offset is minutes east of UTC."""
    if (start is None) != (end is None):
        raise HTTPException(status_code=400, detail="start and end must be given together")
    if start is not None and end is not None and end <= start:
        raise HTTPException(status_code=400, detail="end must be after start")
    return AnalyticsFilters(start=start, end=end, tz_offset_min=tz_offset, method=method or None,
                            verdict=verdict, path=path or None, source=source)


@router.get("/overview")
def analytics_overview(
    start: Optional[float] = Query(None), end: Optional[float] = Query(None), tz_offset: int = Query(0, ge=-840, le=840),
    method: Optional[str] = Query(None), verdict: Optional[Literal["anomaly", "normal"]] = Query(None),
    path: Optional[str] = Query(None), source: Optional[Literal["direct", "batch", "replay"]] = Query(None),
) -> Dict[str, object]:
    """KPIs, time series, distributions, score statistics, top endpoints, URL-classifier activity and
    rule-based insights for one date range, compared with the equal-length period before it."""
    return _svc().overview(_filters(start, end, tz_offset, method, verdict, path, source))


@router.get("/breakdown")
def analytics_breakdown(
    start: Optional[float] = Query(None), end: Optional[float] = Query(None), tz_offset: int = Query(0, ge=-840, le=840),
    method: Optional[str] = Query(None), verdict: Optional[Literal["anomaly", "normal"]] = Query(None),
    path: Optional[str] = Query(None), source: Optional[Literal["direct", "batch", "replay"]] = Query(None),
    group_by: GroupBy = Query("path"), sort: SortKey = Query("anomalies"), order: Literal["asc", "desc"] = Query("desc"),
    search: Optional[str] = Query(None, description="Group key contains"),
    limit: int = Query(25, ge=1, le=500), offset: int = Query(0, ge=0),
) -> Dict[str, object]:
    """Per-endpoint / per-client / per-method / per-source aggregates (server-side sort, search, paging)."""
    f = _filters(start, end, tz_offset, method, verdict, path, source)
    return _svc().breakdown(f, group_by, sort, order, search or None, limit, offset)


@router.get("/export")
def analytics_export(
    start: Optional[float] = Query(None), end: Optional[float] = Query(None), tz_offset: int = Query(0, ge=-840, le=840),
    method: Optional[str] = Query(None), verdict: Optional[Literal["anomaly", "normal"]] = Query(None),
    path: Optional[str] = Query(None), source: Optional[Literal["direct", "batch", "replay"]] = Query(None),
    group_by: GroupBy = Query("path"), sort: SortKey = Query("anomalies"), order: Literal["asc", "desc"] = Query("desc"),
    search: Optional[str] = Query(None),
) -> Response:
    """The full breakdown for the current filters as CSV."""
    f = _filters(start, end, tz_offset, method, verdict, path, source)
    body = _svc().breakdown_csv(f, group_by, sort, order, search or None)
    return Response(
        content=body,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="waf-analytics-{group_by}.csv"'},
    )


__all__ = ["router", "configure"]
