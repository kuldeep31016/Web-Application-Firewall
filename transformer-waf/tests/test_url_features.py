from __future__ import annotations

import pytest

from src.phishing.missingness import MissingnessConfig, apply_missingness, realised_missing_rate, select_missing
from src.phishing.url_features import (
    FEATURE_NAMES,
    URLValidationError,
    compose_url_text,
    normalize_feature_names,
    split_url,
    validate_url,
)
from src.preprocessing.tokenizer import MISSING_TOKEN, HTTPRequestTokenizer


# ---- input validation --------------------------------------------------------

def test_valid_url_passes():
    assert validate_url("https://example.com/products?id=123") == "https://example.com/products?id=123"


@pytest.mark.parametrize("bad", ["", "   ", "http://", "has space.com", "a\tb.com", "x" * 3000])
def test_invalid_urls_are_rejected(bad):
    with pytest.raises(URLValidationError):
        validate_url(bad)


def test_non_string_rejected():
    with pytest.raises(URLValidationError):
        validate_url(123)  # type: ignore[arg-type]


def test_very_long_but_valid_url():
    url = "https://example.com/" + "a" * 1500 + "?q=" + "b" * 400
    assert validate_url(url) == url


def test_special_characters_in_query():
    seg = split_url("https://example.com/search?q=<script>alert(1)</script>&x=%27")
    assert seg.query == "q=<script>alert(1)</script>&x=%27"
    assert seg.domain == "example"


# ---- segmentation ------------------------------------------------------------

def test_split_full_url():
    seg = split_url("https://login.secure-paypal.co.uk:8443/account/verify.php?id=1&ref=mail#top")
    assert seg.as_dict() == {
        "scheme": "https",
        "subdomain": "login",
        "domain": "secure-paypal",
        "tld": "co.uk",
        "path": "/account/verify.php",
        "query": "id=1&ref=mail",
        "fragment": "top",
    }
    assert seg.port == "8443"


def test_split_ip_host_and_no_scheme():
    seg = split_url("192.168.0.1/x")
    assert seg.scheme == "" and seg.domain == "192.168.0.1" and seg.tld == "" and seg.path == "/x"


def test_compose_roundtrip_without_missing():
    url = "https://www.uni-mainz.de/path?a=1#frag"
    assert compose_url_text(split_url(url)) == url


# ---- missingness -------------------------------------------------------------

def test_missing_token_substitution_and_blank():
    seg = split_url("https://www.example.com/login?user=test")
    text = compose_url_text(seg, {"scheme", "domain"})
    assert text == f"{MISSING_TOKEN}://www.{MISSING_TOKEN}.com/login?user=test"
    blank = compose_url_text(seg, {"scheme", "domain"}, representation="blank")
    assert blank == "www.com/login?user=test"
    # removed information cannot be found anywhere in the composed text
    assert "example" not in text and "https" not in text


def test_all_features_missing():
    seg = split_url("https://www.example.com/login?user=test#x")
    text = compose_url_text(seg, set(FEATURE_NAMES))
    assert "example" not in text and "login" not in text and "user" not in text
    assert text.count(MISSING_TOKEN) == len(FEATURE_NAMES)
    assert compose_url_text(seg, set(FEATURE_NAMES), representation="blank") == ""


def test_invalid_feature_names():
    with pytest.raises(ValueError):
        normalize_feature_names(["url_length"])
    with pytest.raises(ValueError):
        MissingnessConfig(features=("nope",))
    with pytest.raises(ValueError):
        MissingnessConfig(rate=1.5)
    with pytest.raises(ValueError):
        MissingnessConfig(representation="average")


def test_no_missing_features_is_identity():
    segs = [split_url("https://a.com/x"), split_url("http://b.org/y?z=1")]
    out = apply_missingness(segs, MissingnessConfig())
    assert [o.text for o in out] == ["https://a.com/x", "http://b.org/y?z=1"]
    assert realised_missing_rate(out) == 0.0


def test_rate_mode_is_seeded_and_close_to_rate():
    n = 4000
    cfg = MissingnessConfig(rate=0.3, seed=7)
    a = select_missing(cfg, n)
    b = select_missing(cfg, n)
    assert a == b  # reproducible
    realised = sum(len(s) for s in a) / (n * len(FEATURE_NAMES))
    assert abs(realised - 0.3) < 0.03
    c = select_missing(MissingnessConfig(rate=0.3, seed=8), n)
    assert c != a  # different seed, different draw


def test_fixed_features_always_removed_in_rate_mode():
    cfg = MissingnessConfig(features=("scheme",), rate=0.2, seed=1)
    for s in select_missing(cfg, 50):
        assert "scheme" in s


# ---- tokenizer marker --------------------------------------------------------

def test_missing_token_is_atomic_in_tokenizer():
    tok = HTTPRequestTokenizer(vocab_size=100)
    tok.build_vocab(["GET /a?b=c"])
    toks = tok._basic_tokenize(f"{MISSING_TOKEN}://x.com/{MISSING_TOKEN}")
    assert toks.count(MISSING_TOKEN) == 2
    assert MISSING_TOKEN in tok.token_to_id
    assert tok.tokenize("GET /a") == tok.tokenize("GET /a")  # deterministic
