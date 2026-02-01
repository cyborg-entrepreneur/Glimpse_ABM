"""Utility helpers and AI label normalization for Glimpse ABM."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np


def safe_mean(data: Any) -> float:
    """Compute the mean, returning NaN for empty collections."""
    arr = np.asarray(data, dtype=float)
    if arr.size == 0:
        return np.nan
    with np.errstate(invalid="ignore"):
        return float(arr.mean())


def fast_mean(values: Iterable[float]) -> float:
    """Lightweight mean for Python iterables; matches NumPy for finite inputs."""
    if isinstance(values, np.ndarray):
        if values.size == 0:
            return float("nan")
        with np.errstate(invalid="ignore"):
            return float(values.mean())
    total = 0.0
    count = 0
    for value in values:
        total += float(value)
        count += 1
    if count == 0:
        return float("nan")
    return total / count


def safe_exp(value: Any, limit: float = 4.0) -> Any:
    """Exponentiate while preventing overflow through smooth saturation."""
    arr = np.asarray(value, dtype=float)
    scaled = arr / (1.0 + np.abs(arr) / max(limit, 1e-6))
    clipped = np.clip(scaled, -limit, limit)
    result = np.exp(clipped)
    if np.isscalar(value):
        return float(result)
    return result


def stable_sigmoid(x: Any) -> Any:
    """Numerically stable logistic function without hard clipping."""
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        val = float(arr)
        if val >= 0:
            return float(1.0 / (1.0 + np.exp(-val)))
        exp_val = np.exp(val)
        return float(exp_val / (1.0 + exp_val))
    result = np.empty_like(arr, dtype=float)
    positive_mask = arr >= 0
    if np.any(positive_mask):
        result[positive_mask] = 1.0 / (1.0 + np.exp(-arr[positive_mask]))
    negative_mask = ~positive_mask
    if np.any(negative_mask):
        exp_x = np.exp(arr[negative_mask])
        result[negative_mask] = exp_x / (1.0 + exp_x)
    if np.isscalar(x):
        return float(result)
    return result


AI_LEVEL_NORMALIZATION = {
    "none": "none",
    "no_ai": "none",
    "human": "none",
    "human_only": "none",
    "human_agent": "none",
    "manual": "none",
    "basic": "basic",
    "basic_ai": "basic",
    "basic-ai": "basic",
    "basic_ai_level": "basic",
    "advanced": "advanced",
    "advanced_ai": "advanced",
    "advanced-ai": "advanced",
    "premium": "premium",
    "premium_ai": "premium",
    "premium-ai": "premium",
}

AI_CANONICAL_TO_DISPLAY = {
    "none": "human",
    "basic": "basic_ai",
    "advanced": "advanced_ai",
    "premium": "premium_ai",
}

DISPLAY_TO_CANONICAL = {
    display: canonical for canonical, display in AI_CANONICAL_TO_DISPLAY.items()
}


def normalize_ai_label(value: Any, default: str = "none") -> str:
    """Map varied AI level labels to canonical categories."""
    if value is None:
        return default
    try:
        key = str(value).strip().lower()
    except Exception:
        key = str(value).lower()
    key = key.replace("-", "_").replace(" ", "_")
    if key in AI_LEVEL_NORMALIZATION:
        return AI_LEVEL_NORMALIZATION[key]
    if key in AI_CANONICAL_TO_DISPLAY:
        return key
    if key in DISPLAY_TO_CANONICAL:
        return DISPLAY_TO_CANONICAL[key]
    return default


def canonical_to_display(level: str, default: str = "human") -> str:
    """Convert canonical AI level into display label used in figures."""
    if level in AI_CANONICAL_TO_DISPLAY:
        return AI_CANONICAL_TO_DISPLAY[level]
    if level in DISPLAY_TO_CANONICAL:
        return level
    return default
