"""Validation and normalisation of prediction requests."""
import math
from datetime import datetime
from typing import Any, Dict, Mapping, Optional

from delivery.config import CATEGORICAL_INPUTS, CLOCK_FIELDS, NUMERIC_INPUTS, NumericSpec


class ValidationError(ValueError):
    """Raised when a request contains invalid fields; ``errors`` maps each
    offending field to a human-readable message."""

    def __init__(self, errors: Dict[str, str]):
        super().__init__("; ".join(f"{k}: {v}" for k, v in errors.items()))
        self.errors = errors


def _canonical_key(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


# "semi urban", "Semi_Urban" and "semi-urban" all resolve to "Semi-Urban".
_CATEGORY_LOOKUP = {
    field: {_canonical_key(level): level for level in levels}
    for field, (levels, _default) in CATEGORICAL_INPUTS.items()
}


def _parse_number(value: Any, spec: NumericSpec) -> float:
    # bool is a subclass of int; "true" is never a sensible distance.
    if isinstance(value, bool):
        raise ValueError("must be a number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):  # OverflowError: huge JSON ints
        raise ValueError("must be a number") from None
    if not math.isfinite(number):
        raise ValueError("must be a finite number")
    if spec.integer:
        if not number.is_integer():
            raise ValueError("must be a whole number")
        number = int(number)
    if not spec.min <= number <= spec.max:
        raise ValueError(f"must be between {spec.min:g} and {spec.max:g}")
    return number


def _parse_category(field: str, value: Any) -> str:
    if field == "festival" and isinstance(value, bool):
        return "Yes" if value else "No"
    if not isinstance(value, str):
        raise ValueError("must be a string")
    level = _CATEGORY_LOOKUP[field].get(_canonical_key(value))
    if level is None:
        allowed = ", ".join(CATEGORICAL_INPUTS[field][0])
        raise ValueError(f"must be one of: {allowed}")
    return level


def validate_order(payload: Any, now: Optional[datetime] = None) -> Dict[str, Any]:
    """Return a normalised copy of ``payload`` or raise ``ValidationError``.

    Missing optional fields take their documented defaults; ``order_hour`` and
    ``day_of_week`` default to the current time. Unknown extra keys are ignored.
    """
    if not isinstance(payload, Mapping):
        raise ValidationError({"body": "request body must be a JSON object"})

    now = now or datetime.now()
    clock_defaults = {"order_hour": now.hour, "day_of_week": now.weekday()}
    order: Dict[str, Any] = {}
    errors: Dict[str, str] = {}

    for field, spec in NUMERIC_INPUTS.items():
        value = payload.get(field)
        if value is None or value == "":
            if field in CLOCK_FIELDS:
                order[field] = clock_defaults[field]
            elif spec.default is None:
                errors[field] = "is required"
            else:
                order[field] = spec.default
            continue
        try:
            order[field] = _parse_number(value, spec)
        except ValueError as exc:
            errors[field] = str(exc)

    for field, (_levels, default) in CATEGORICAL_INPUTS.items():
        value = payload.get(field)
        if value is None or value == "":
            order[field] = default
            continue
        try:
            order[field] = _parse_category(field, value)
        except ValueError as exc:
            errors[field] = str(exc)

    if errors:
        raise ValidationError(errors)
    return order
