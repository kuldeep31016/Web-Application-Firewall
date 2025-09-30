from __future__ import annotations

import os
import time
import uuid
from typing import Dict, List

from fastapi import FastAPI, HTTPException, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from fastapi.security.api_key import APIKeyHeader

from ..models.inference import InferenceEngine
from ..preprocessing.tokenizer import HTTPRequestTokenizer
from ..utils.config import load_config, CONFIG_PATH
from ..utils.logger import logger, setup_logging


class DetectionRequest(BaseModel):
    method: str
    path: str
    query_params: Dict[str, str]
    headers: Dict[str, str]
    body: str = ""


class DetectionResponse(BaseModel):
    is_anomaly: bool
    confidence: float
    anomaly_score: float
    timestamp: str
    request_id: str


app = FastAPI(title="WAF Detection API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


CONFIG = load_config(CONFIG_PATH)
ENGINE: InferenceEngine | None = None
TOKENIZER: HTTPRequestTokenizer | None = None
API_KEY = os.environ.get("WAF_API_KEY", "dev-key")
METRICS = {"requests": 0, "anomalies": 0}
_RATE_WINDOW_S = 60
_RATE_LIMIT = 1000
_RATE_BUCKET: dict[str, list[float]] = {}

# Swagger-friendly API key (enables Authorize button)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str | None = Security(api_key_header)) -> bool:
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return True


def _auth(request: Request) -> None:
    key = request.headers.get("x-api-key") or request.headers.get("X-API-Key")
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


def _rate_limit(request: Request) -> None:
    now = time.time()
    ip = request.client.host if request.client else "unknown"
    bucket = _RATE_BUCKET.setdefault(ip, [])
    # Drop old
    threshold = now - _RATE_WINDOW_S
    while bucket and bucket[0] < threshold:
        bucket.pop(0)
    if len(bucket) >= _RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    bucket.append(now)


@app.on_event("startup")
async def load_model() -> None:  # noqa: D401
    setup_logging(os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "logs", "detection_logs"))
    global ENGINE, TOKENIZER
    threshold = float(CONFIG.get("detection", {}).get("threshold", 0.75))
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "models", "checkpoints", "best.pt")
    ENGINE = InferenceEngine(model_path=model_path, threshold=threshold)
    try:
        ENGINE.load_model()
    except FileNotFoundError:
        logger.warning("Model checkpoint not found, API will return dummy responses until trained.")
        ENGINE.model = None  # type: ignore[assignment]
    TOKENIZER = HTTPRequestTokenizer(vocab_size=int(CONFIG.get("model", {}).get("vocab_size", 10000)))


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics")
async def metrics() -> Dict[str, int]:
    return dict(METRICS)


@app.post("/threshold")
async def update_threshold(value: float, _: bool = Security(verify_api_key)) -> Dict[str, float]:
    global ENGINE
    if ENGINE is None:
        raise HTTPException(status_code=503, detail="Engine not ready")
    ENGINE.update_threshold(value)
    return {"threshold": value}


def _encode_request(payload: DetectionRequest, max_len: int) -> Dict[str, List[int]]:
    assert TOKENIZER is not None
    composed = f"{payload.method} {payload.path}?" + "&".join(f"{k}={v}" for k, v in sorted(payload.query_params.items()))
    if payload.body:
        composed += " BODY:" + payload.body
    return TOKENIZER.encode(composed, max_length=max_len)


@app.post("/detect", response_model=DetectionResponse)
async def detect_anomaly(payload: DetectionRequest, request: Request, _: bool = Security(verify_api_key)) -> DetectionResponse:
    _rate_limit(request)
    global ENGINE
    if ENGINE is None:
        raise HTTPException(status_code=503, detail="Engine not ready")
    METRICS["requests"] += 1
    if ENGINE.model is None:
        # Fallback behavior when model missing: mark as not anomaly
        rid = str(uuid.uuid4())
        return DetectionResponse(is_anomaly=False, confidence=0.0, anomaly_score=0.0, timestamp=str(time.time()), request_id=rid)
    max_len_cfg = int(CONFIG.get("model", {}).get("max_seq_length", 512))
    if ENGINE.model is not None:
        # Cap by model's positional embedding size
        max_len_model = int(getattr(getattr(ENGINE.model, "positional", None), "num_embeddings", max_len_cfg))
        max_len = min(max_len_cfg, max_len_model)
    else:
        max_len = max_len_cfg
    encoded = _encode_request(payload, max_len)
    result = ENGINE.predict_single(encoded["input_ids"], encoded["attention_mask"])
    rid = str(uuid.uuid4())
    if result["is_anomaly"]:  # type: ignore[index]
        METRICS["anomalies"] += 1
    return DetectionResponse(
        is_anomaly=bool(result["is_anomaly"]),  # type: ignore[index]
        confidence=float(result["confidence"]),  # type: ignore[index]
        anomaly_score=float(result["anomaly_score"]),  # type: ignore[index]
        timestamp=str(time.time()),
        request_id=rid,
    )


@app.post("/detect/batch")
async def detect_batch(requests: List[DetectionRequest], request: Request, _: bool = Security(verify_api_key)):
    _rate_limit(request)
    global ENGINE
    if ENGINE is None:
        raise HTTPException(status_code=503, detail="Engine not ready")
    METRICS["requests"] += len(requests)
    if ENGINE.model is None:
        return [
            {
                "is_anomaly": False,
                "confidence": 0.0,
                "anomaly_score": 0.0,
                "timestamp": str(time.time()),
                "request_id": str(uuid.uuid4()),
            }
            for _ in requests
        ]
    max_len_cfg = int(CONFIG.get("model", {}).get("max_seq_length", 512))
    if ENGINE.model is not None:
        max_len_model = int(getattr(getattr(ENGINE.model, "positional", None), "num_embeddings", max_len_cfg))
        max_len = min(max_len_cfg, max_len_model)
    else:
        max_len = max_len_cfg
    encoded_inputs = []
    encoded_masks = []
    for p in requests:
        enc = _encode_request(p, max_len)
        encoded_inputs.append(enc["input_ids"])  # type: ignore[index]
        encoded_masks.append(enc["attention_mask"])  # type: ignore[index]
    results = ENGINE.predict_batch(encoded_inputs, encoded_masks)
    out = []
    for r in results:
        if r["is_anomaly"]:  # type: ignore[index]
            METRICS["anomalies"] += 1
        out.append(
            {
                "is_anomaly": bool(r["is_anomaly"]),  # type: ignore[index]
                "confidence": float(r["confidence"]),  # type: ignore[index]
                "anomaly_score": float(r["anomaly_score"]),  # type: ignore[index]
                "timestamp": str(time.time()),
                "request_id": str(uuid.uuid4()),
            }
        )
    return out


