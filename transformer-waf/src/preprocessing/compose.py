"""
Canonical text composition for HTTP requests.

The Transformer operates on a single token sequence per request. Every code
path that feeds the model (detection API, training data preparation, the
URL research module) must build that string the same way, so the logic lives
here rather than being duplicated.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional


def compose_request_text(
    method: str,
    path: str,
    query_params: Optional[Mapping[str, str]] = None,
    body: str = "",
) -> str:
    """Compose the model-facing text for a request.

    Format (unchanged from the original detection API):
        "{METHOD} {path}?k1=v1&k2=v2 BODY:{body}"

    Query parameters are sorted by key so the representation is deterministic.
    The trailing "?" is kept even when there are no parameters to stay
    byte-compatible with the representation the existing model was trained on.
    """
    params = query_params or {}
    composed = f"{method} {path}?" + "&".join(f"{k}={v}" for k, v in sorted(params.items()))
    if body:
        composed += " BODY:" + body
    return composed


def parse_flat_request_line(line: str) -> Dict[str, object]:
    """Parse the flat ``"METHOD /path k=v&k2=v2 {json body}"`` format used by
    the raw datasets in ``data/training/``.

    Returns a dict with ``method``, ``path``, ``query_params`` and ``body``.
    """
    parts = line.strip().split(" ", 2)
    method = parts[0].upper() if parts else "GET"
    path = parts[1] if len(parts) > 1 else "/"
    rest = parts[2] if len(parts) > 2 else ""
    query_params: Dict[str, str] = {}
    body = ""
    if rest:
        # A query string is the first whitespace-delimited chunk containing '='
        chunk, _, remainder = rest.partition(" ")
        if "=" in chunk and not chunk.startswith("{"):
            for pair in chunk.split("&"):
                key, _, value = pair.partition("=")
                if key:
                    query_params[key] = value
            body = remainder.strip()
        else:
            body = rest.strip()
    return {"method": method, "path": path, "query_params": query_params, "body": body}


__all__ = ["compose_request_text", "parse_flat_request_line"]
