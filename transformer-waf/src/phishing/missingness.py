"""
Controlled missing-information simulation.

Two modes are supported, mirroring the research design:

* **Fixed features** - a chosen set of information units is unavailable for
  every sample (single-feature and multi-feature ablation).
* **Missing rate** - each information unit of each sample is independently
  unavailable with probability ``rate`` (Bernoulli, seeded). The expected
  fraction of units removed per URL therefore equals ``rate``; the realised
  fraction is reported alongside the results.

Both modes can be combined: fixed features are always removed and the rate
applies to the remaining units.

Every call is deterministic for a given seed, so experiments are reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from .url_features import FEATURE_NAMES, URLSegments, compose_url_text, normalize_feature_names


REPRESENTATIONS = ("blank", "missing_token")


@dataclass(frozen=True)
class MissingnessConfig:
    """Describes which information is unavailable and how it is represented."""

    features: Tuple[str, ...] = ()
    rate: float = 0.0
    representation: str = "missing_token"
    seed: int = 42

    def __post_init__(self) -> None:
        normalize_feature_names(self.features)
        if not 0.0 <= float(self.rate) <= 1.0:
            raise ValueError("rate must be between 0.0 and 1.0")
        if self.representation not in REPRESENTATIONS:
            raise ValueError(f"representation must be one of {REPRESENTATIONS}")

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "MissingnessConfig":
        feats = data.get("features") or ()
        return cls(
            features=tuple(sorted(normalize_feature_names(feats))),  # type: ignore[arg-type]
            rate=float(data.get("rate", 0.0) or 0.0),
            representation=str(data.get("representation", "missing_token") or "missing_token"),
            seed=int(data.get("seed", 42) or 42),
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "features": list(self.features),
            "rate": self.rate,
            "representation": self.representation,
            "seed": self.seed,
        }

    @property
    def is_complete(self) -> bool:
        return not self.features and self.rate == 0.0


@dataclass
class MaskedSample:
    text: str
    missing: Set[str]
    present_before: List[str] = field(default_factory=list)

    @property
    def removed_present(self) -> Set[str]:
        """Units that were present in the URL and have actually been removed."""
        return self.missing & set(self.present_before)


def select_missing(config: MissingnessConfig, n_samples: int) -> List[Set[str]]:
    """Draw the per-sample set of unavailable units. Deterministic for a seed."""
    fixed = set(config.features)
    if config.rate <= 0.0:
        return [set(fixed) for _ in range(n_samples)]
    rng = np.random.RandomState(config.seed)
    draws = rng.random_sample((n_samples, len(FEATURE_NAMES))) < config.rate
    out: List[Set[str]] = []
    for row in draws:
        chosen = {name for name, hit in zip(FEATURE_NAMES, row) if hit}
        out.append(chosen | fixed)
    return out


def apply_missingness(segments_list: Sequence[URLSegments], config: MissingnessConfig) -> List[MaskedSample]:
    """Remove information from every sample according to ``config`` and compose
    the model-facing text from what remains."""
    missing_sets = select_missing(config, len(segments_list))
    out: List[MaskedSample] = []
    for segments, missing in zip(segments_list, missing_sets):
        text = compose_url_text(segments, missing, representation=config.representation)
        out.append(MaskedSample(text=text, missing=missing, present_before=segments.present_features()))
    return out


def realised_missing_rate(samples: Iterable[MaskedSample]) -> float:
    """Fraction of information units (over all samples) that were removed."""
    samples = list(samples)
    if not samples:
        return 0.0
    total_units = len(samples) * len(FEATURE_NAMES)
    removed = sum(len(s.missing) for s in samples)
    return removed / total_units


__all__ = [
    "REPRESENTATIONS",
    "MissingnessConfig",
    "MaskedSample",
    "select_missing",
    "apply_missingness",
    "realised_missing_rate",
]
