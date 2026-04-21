"""Helpers for normalized annotation confidence handling."""

from __future__ import annotations

from collections.abc import Mapping


CONFIDENCE_RANK = {
    "low": 0,
    "medium": 1,
    "high": 2,
}

CONFIDENCE_ALIASES = {
    "certain": "high",
    "uncertain": "low",
}


def normalize_confidence(value, default: str = "high") -> str:
    """Normalize confidence labels into low/medium/high."""
    if value is None:
        return default
    text = str(value).strip().lower()
    if not text or text == "nan":
        return default
    text = CONFIDENCE_ALIASES.get(text, text)
    if text in CONFIDENCE_RANK:
        return text
    return default


def confidence_at_least(value, minimum: str = "low", default: str = "high") -> bool:
    """Return whether a confidence value passes the requested minimum."""
    normalized = normalize_confidence(value, default=default)
    threshold = normalize_confidence(minimum, default="low")
    return CONFIDENCE_RANK[normalized] >= CONFIDENCE_RANK[threshold]


def confidence_weight(
    value,
    weights: Mapping[str, float] | None = None,
    default_confidence: str = "high",
) -> float:
    """Map confidence labels to numeric training weights."""
    normalized = normalize_confidence(value, default=default_confidence)
    weights = weights or {"low": 0.5, "medium": 0.75, "high": 1.0}
    return float(weights.get(normalized, weights.get("high", 1.0)))
