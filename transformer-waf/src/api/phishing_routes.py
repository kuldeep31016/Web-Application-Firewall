"""
Research-module routes, mounted on the existing WAF Detection API
(same port, same X-API-Key header). All results are computed by
``src.phishing.service.PhishingService``; nothing here is simulated.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from ..phishing.dataset import DatasetError
from ..phishing.experiments import DEFAULT_RATES, MAX_SAMPLE_SIZE, ExperimentError
from ..phishing.service import PhishingService
from ..phishing.url_features import URLValidationError


# Authentication is applied by the detection API when it includes this router
# (``dependencies=[Security(verify_api_key)]``), so every route here requires
# the same X-API-Key as the existing endpoints.
router = APIRouter()
_service: Optional[PhishingService] = None


def configure(service: PhishingService) -> None:
    global _service
    _service = service


def _svc() -> PhishingService:
    if _service is None:
        raise HTTPException(status_code=503, detail="Phishing service not configured")
    return _service


class URLDetectionRequest(BaseModel):
    url: str = Field(..., description="URL to analyse (not fetched, only parsed)")
    missing_features: List[str] = Field(default_factory=list, description="Information units to treat as unavailable")
    strategy: str = Field("none", description="none | missing_token | augmented_training")
    persist: bool = Field(True, description="Store the analysis in the URL history")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"url": "https://www.readersdigest.co.uk", "missing_features": [], "strategy": "none", "persist": True},
                {"url": "https://www.readersdigest.co.uk", "missing_features": ["subdomain", "scheme"], "strategy": "augmented_training", "persist": False},
                {"url": "http://secure-login.paypa1-verify.tk/account/update.php?id=99", "missing_features": ["domain"], "strategy": "none", "persist": True},
            ]
        }
    }


class MissingInformationRequest(BaseModel):
    rates: List[float] = Field(default_factory=lambda: list(DEFAULT_RATES), description="Missing-information levels (0-1)")
    features: List[str] = Field(default_factory=list, description="Information units always missing (single/multi-feature conditions)")
    strategies: Optional[List[str]] = Field(None, description="Subset of handling strategies; default all")
    sample_size: Optional[int] = Field(5000, ge=10, le=MAX_SAMPLE_SIZE, description="Test URLs to evaluate (stratified, seeded); null = whole test split")
    seed: int = Field(42)


class FeatureImpactRequest(BaseModel):
    features: Optional[List[str]] = Field(None, description="Units to ablate individually; default all")
    combinations: List[List[str]] = Field(default_factory=list, description="Extra feature combinations to ablate together")
    strategies: Optional[List[str]] = Field(None)
    sample_size: Optional[int] = Field(5000, ge=10, le=MAX_SAMPLE_SIZE)
    seed: int = Field(42)


def _wrap(fn):
    """Translate domain errors into HTTP errors without hiding them."""
    try:
        return fn()
    except URLValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except (ValueError, ExperimentError, DatasetError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/url/status", tags=["url"])
def url_status() -> Dict[str, object]:
    """Model, dataset, feature and strategy availability for the research module."""
    return _svc().status()


@router.post("/detect/url", tags=["url"])
def detect_url(payload: URLDetectionRequest) -> Dict[str, object]:
    return _wrap(lambda: _svc().analyze_url(payload.url, payload.missing_features, payload.strategy, persist=payload.persist))


@router.get("/url/summary", tags=["url"])
def url_summary() -> Dict[str, object]:
    """Counts of stored URL analyses (for the overview page)."""
    return _svc().store.url_detection_stats()


@router.get("/url/history", tags=["url"])
def url_history(limit: int = Query(100, ge=1, le=1000), offset: int = Query(0, ge=0)) -> List[Dict[str, object]]:
    return _svc().store.get_url_detections(limit=limit, offset=offset)


@router.post("/experiment/missing-information", tags=["experiment"])
def experiment_missing_information(payload: MissingInformationRequest) -> Dict[str, object]:
    return _wrap(
        lambda: _svc().run_missing_information(
            rates=payload.rates,
            features=payload.features,
            strategies=payload.strategies,
            sample_size=payload.sample_size,
            seed=payload.seed,
        )
    )


@router.post("/experiment/feature-impact", tags=["experiment"])
def experiment_feature_impact(payload: FeatureImpactRequest) -> Dict[str, object]:
    return _wrap(
        lambda: _svc().run_feature_impact(
            features=payload.features,
            combinations=payload.combinations,
            strategies=payload.strategies,
            sample_size=payload.sample_size,
            seed=payload.seed,
        )
    )


@router.get("/experiment/results", tags=["experiment"])
def experiment_results(limit: int = Query(50, ge=1, le=500), kind: Optional[str] = Query(None)) -> List[Dict[str, object]]:
    return _svc().store.list_experiments(limit=limit, kind=kind)


@router.get("/experiment/results/{experiment_id}", tags=["experiment"])
def experiment_result(experiment_id: str) -> Dict[str, object]:
    exp = _svc().store.get_experiment(experiment_id)
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return exp


@router.delete("/experiment/results/{experiment_id}", tags=["experiment"])
def delete_experiment(experiment_id: str) -> Dict[str, bool]:
    return {"deleted": _svc().store.delete_experiment(experiment_id)}


@router.post("/url/reload", tags=["url"])
def reload_models() -> Dict[str, bool]:
    return _svc().reload()


__all__ = ["router", "configure"]
