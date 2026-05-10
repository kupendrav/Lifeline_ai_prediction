"""
LIFELINE AI — Input Validation & Sanitisation Service
All user inputs are strictly validated before reaching the prediction engine.
"""
from __future__ import annotations
from typing import Any, Dict, Tuple


VALID_CATEGORICALS = {
    "sex":        {"female", "male", "other"},
    "education":  {"less_hs", "hs", "some_college", "bachelors", "graduate"},
    "ses":        {"low", "lower_middle", "middle", "upper_middle", "high"},
    "smoking":    {"never", "former", "light", "moderate", "heavy"},
    "alcohol":    {"never", "light", "moderate", "heavy"},
    "processed":  {"rarely", "sometimes", "often", "very_often"},
    "conditions": {"none", "hypertension", "diabetes", "heart_disease", "multiple"},
    "family":     {"below_70", "70_80", "80_90", "above_90"},
    "mental":     {"excellent", "good", "moderate", "poor"},
    "env":        {"rural", "suburban", "urban", "high_pollution"},
}

NUMERIC_RANGES = {
    "age":      (18, 100),
    "exercise": (0, 7),
    "sleep":    (3.0, 12.0),
    "stress":   (1, 10),
    "social":   (1, 10),
    "fv":       (0, 14),
    "bmi":      (12.0, 60.0),
    "hr":       (30, 130),
    "bp":       (70, 200),
    "aqi":      (0, 500),
}


def _coerce_numeric(key: str, raw: Any) -> float:
    """Cast to float and clamp to valid range."""
    try:
        val = float(raw)
    except (TypeError, ValueError):
        lo, hi = NUMERIC_RANGES[key]
        return (lo + hi) / 2.0
    lo, hi = NUMERIC_RANGES[key]
    return max(lo, min(hi, val))


def validate_and_clean(raw: Dict[str, Any]) -> Tuple[Dict[str, Any], list]:
    """
    Returns (cleaned_dict, error_list).
    Errors are non-fatal (clamped to defaults); returned for logging only.
    """
    cleaned = {}
    errors  = []

    for key, (lo, hi) in NUMERIC_RANGES.items():
        cleaned[key] = _coerce_numeric(key, raw.get(key))

    for key, valid_set in VALID_CATEGORICALS.items():
        val = str(raw.get(key, "")).strip().lower()
        if val in valid_set:
            cleaned[key] = val
        else:
            # fallback to first alphabetically-sorted default
            default = sorted(valid_set)[len(valid_set) // 2]
            cleaned[key] = default
            errors.append(f"Invalid value '{val}' for '{key}'; defaulted to '{default}'")

    return cleaned, errors
