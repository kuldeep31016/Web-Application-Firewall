"""
URL information units ("features") for the robustness experiments.

The Transformer in this project does not consume a tabular feature vector; it
consumes a token sequence. The units of information that can genuinely be
"unavailable" are therefore the structural segments of the URL that the
sequence is composed from:

    scheme      http / https
    subdomain   everything left of the registrable domain (e.g. "www", "login.secure")
    domain      the registrable label (e.g. "paypal-verify")
    tld         public suffix (e.g. "com", "co.uk")
    path        "/account/verify.php"
    query       "id=1&ref=mail"
    fragment    "#section"

Missing information is simulated by removing one or more of these segments
from the text before tokenisation. Two representations are supported:

    blank          the segment and its separators are dropped, as if it had
                   never been extracted
    missing_token  the segment is replaced by the explicit ``[MISSING]`` marker

Nothing in this module can reconstruct a removed segment: the composed text
is built only from the segments that remain.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set
from urllib.parse import urlsplit

from ..preprocessing.tokenizer import MISSING_TOKEN


FEATURE_NAMES: List[str] = ["scheme", "subdomain", "domain", "tld", "path", "query", "fragment"]

FEATURE_DESCRIPTIONS: Dict[str, str] = {
    "scheme": "Protocol (http or https)",
    "subdomain": "Host labels left of the registrable domain (e.g. www, login.secure)",
    "domain": "Registrable domain label (e.g. paypal-verify)",
    "tld": "Public suffix / top-level domain (e.g. com, co.uk)",
    "path": "Resource path (e.g. /account/verify.php)",
    "query": "Query string parameters",
    "fragment": "Fragment identifier after #",
}

MAX_URL_LENGTH = 2048

# Common second-level public suffixes so "example.co.uk" splits as domain=example, tld=co.uk
_SECOND_LEVEL_SUFFIXES = {
    "co.uk", "org.uk", "ac.uk", "gov.uk", "com.au", "net.au", "org.au", "co.nz", "co.in", "net.in", "org.in",
    "co.jp", "ne.jp", "or.jp", "com.br", "com.mx", "com.ar", "co.za", "com.tr", "com.sg", "com.my", "co.kr",
    "com.cn", "net.cn", "org.cn", "com.hk", "com.tw", "com.ua", "com.pk", "com.ng", "com.eg", "com.sa",
}
_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://")
_IPV4_RE = re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")


class URLValidationError(ValueError):
    """Raised for input that cannot be analysed as a URL."""


@dataclass
class URLSegments:
    """Structural decomposition of a URL into information units."""

    scheme: str = ""
    subdomain: str = ""
    domain: str = ""
    tld: str = ""
    port: str = ""
    path: str = ""
    query: str = ""
    fragment: str = ""
    original: str = ""

    def get(self, name: str) -> str:
        return str(getattr(self, name))

    def present_features(self) -> List[str]:
        return [f for f in FEATURE_NAMES if self.get(f)]

    def as_dict(self) -> Dict[str, str]:
        return {f: self.get(f) for f in FEATURE_NAMES}


def validate_url(url: object) -> str:
    """Validate raw user input and return the stripped URL string."""
    if not isinstance(url, str):
        raise URLValidationError("URL must be a string")
    text = url.strip()
    if not text:
        raise URLValidationError("URL is empty")
    if len(text) > MAX_URL_LENGTH:
        raise URLValidationError(f"URL is longer than {MAX_URL_LENGTH} characters")
    if _CONTROL_RE.search(text) or any(ch.isspace() for ch in text):
        raise URLValidationError("URL contains whitespace or control characters")
    segments = split_url(text)
    if not (segments.domain or segments.subdomain):
        raise URLValidationError("URL has no host")
    return text


def _split_host(host: str) -> Dict[str, str]:
    host = host.strip().lower().rstrip(".")
    if not host:
        return {"subdomain": "", "domain": "", "tld": ""}
    if _IPV4_RE.match(host) or host.startswith("["):
        # IP hosts have no subdomain/tld structure; keep the address as the domain
        return {"subdomain": "", "domain": host, "tld": ""}
    labels = host.split(".")
    if len(labels) == 1:
        return {"subdomain": "", "domain": labels[0], "tld": ""}
    if len(labels) >= 3 and ".".join(labels[-2:]) in _SECOND_LEVEL_SUFFIXES:
        tld = ".".join(labels[-2:])
        domain = labels[-3]
        sub = labels[:-3]
    else:
        tld = labels[-1]
        domain = labels[-2]
        sub = labels[:-2]
    return {"subdomain": ".".join(sub), "domain": domain, "tld": tld}


def split_url(url: str) -> URLSegments:
    """Decompose a URL string into segments. Never raises for odd input."""
    text = url.strip()
    has_scheme = bool(_SCHEME_RE.match(text))
    parsed = urlsplit(text if has_scheme else "//" + text)
    host_parts = _split_host(parsed.hostname or "")
    port = ""
    try:
        port = str(parsed.port) if parsed.port else ""
    except ValueError:
        port = ""
    return URLSegments(
        scheme=parsed.scheme.lower() if has_scheme else "",
        subdomain=host_parts["subdomain"],
        domain=host_parts["domain"],
        tld=host_parts["tld"],
        port=port,
        path=parsed.path,
        query=parsed.query,
        fragment=parsed.fragment,
        original=text,
    )


def normalize_feature_names(features: Optional[Iterable[str]]) -> Set[str]:
    """Validate a user-supplied list of feature names."""
    if not features:
        return set()
    out: Set[str] = set()
    for name in features:
        key = str(name).strip().lower()
        if key not in FEATURE_NAMES:
            raise ValueError(f"Unknown feature {name!r}; valid features: {FEATURE_NAMES}")
        out.add(key)
    return out


def compose_url_text(segments: URLSegments, missing: Optional[Set[str]] = None, representation: str = "missing_token") -> str:
    """Build the model-facing text from the segments that remain available.

    Args:
        segments: decomposed URL.
        missing: feature names that are unavailable.
        representation: "missing_token" substitutes ``[MISSING]``; "blank" drops
            the segment (and its separators) entirely.
    """
    if representation not in ("missing_token", "blank"):
        raise ValueError("representation must be 'missing_token' or 'blank'")
    missing = missing or set()
    marker = MISSING_TOKEN if representation == "missing_token" else ""

    def unit(name: str) -> str:
        return marker if name in missing else segments.get(name)

    scheme = unit("scheme")
    host_labels = [unit("subdomain"), unit("domain"), unit("tld")]
    host = ".".join(label for label in host_labels if label)
    path = unit("path")
    query = unit("query")
    fragment = unit("fragment")

    text = ""
    if scheme:
        text += scheme + "://"
    text += host
    if segments.port:
        text += ":" + segments.port
    if path:
        text += ("/" + path) if path == marker else path
    if query:
        text += "?" + query
    if fragment:
        text += "#" + fragment
    return text


def describe_segments(segments: URLSegments) -> Dict[str, object]:
    """Descriptive information for the UI. These values are derived from the
    segments for display only; the model consumes the token sequence."""
    host = ".".join(x for x in (segments.subdomain, segments.domain, segments.tld) if x)
    return {
        "url_length": len(segments.original),
        "host": host,
        "host_length": len(host),
        "subdomain_count": len(segments.subdomain.split(".")) if segments.subdomain else 0,
        "is_https": segments.scheme == "https",
        "host_is_ip": bool(_IPV4_RE.match(segments.domain)),
        "digit_count": sum(ch.isdigit() for ch in segments.original),
        "query_param_count": len([p for p in segments.query.split("&") if p]) if segments.query else 0,
    }


__all__ = [
    "FEATURE_NAMES",
    "FEATURE_DESCRIPTIONS",
    "MAX_URL_LENGTH",
    "URLSegments",
    "URLValidationError",
    "validate_url",
    "split_url",
    "normalize_feature_names",
    "compose_url_text",
    "describe_segments",
]
