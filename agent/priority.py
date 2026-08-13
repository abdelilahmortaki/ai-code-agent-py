"""Deterministic priority normalization.

Mappings applied, in order:
  1. Canonical form: case-insensitive, surrounding whitespace stripped.
  2. Legacy labels: HIGH -> P2, MEDIUM -> P3, LOW -> P4.
  3. Unknown values are rejected with ValueError (no guessing).
"""

PRIORITIES = ("P1", "P2", "P3", "P4")
LEGACY_PRIORITY_MAP = {"HIGH": "P2", "MEDIUM": "P3", "LOW": "P4"}


def normalize_priority(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"Invalid priority: {value!r} (expected P1, P2, P3 or P4)")
    candidate = value.strip().upper()
    if candidate in LEGACY_PRIORITY_MAP:
        return LEGACY_PRIORITY_MAP[candidate]
    if candidate in PRIORITIES:
        return candidate
    raise ValueError(f"Invalid priority: {value!r} (expected P1, P2, P3 or P4)")
