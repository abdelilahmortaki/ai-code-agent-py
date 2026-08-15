"""FinOps guardrail + consumption trace (F3.3-F3.7).

Deterministic, provider-neutral logic for:
- pre-call token counting (exact when the provider supports it, otherwise a
  documented character-based estimate);
- per-invocation input/context budget thresholds (ticket / project / model);
- bounded context reduction and recount;
- pre-call estimated max cost and post-call actual operational cost estimate
  from *configured* pricing (never invented prices);
- the FinOps STOP outcome when context cannot be reduced under budget.
"""

from __future__ import annotations

import math
from typing import Any, Iterable


class FinOpsStoppedError(ValueError):
    """Generation was stopped before any provider call because the context
    could not be reduced under the effective input budget.

    ``finops`` carries the partial consumption trace so the STOP invocation
    is persisted with the count, budget and threshold outcome that caused it.
    """

    def __init__(self, message: str, finops: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.finops = finops or {}


def estimate_tokens(text: str) -> int:
    """Documented fallback estimate: ceil(chars / 4)."""
    if not text:
        return 0
    return max(0, math.ceil(len(text) / 4))


def count_prompt(provider: Any, prompt: str) -> tuple[int, str]:
    """Count the ACTUAL final prompt; returns (count, count_method).

    Exact when the provider can count tokens, otherwise the documented
    characters/4 estimate. The count method is never silently misreported.
    """
    exact = provider.count_tokens(prompt)
    if exact is not None:
        return int(exact), "exact:provider"
    return estimate_tokens(prompt), "estimate:chars/4"


def compute_effective_budget(limits: Iterable[int | None]) -> int | None:
    """effective_input_budget = min(applicable configured limits)."""
    values = [int(limit) for limit in limits if limit is not None and int(limit) > 0]
    if not values:
        return None
    return min(values)


def evaluate(count: int | None, budget: int | None) -> str:
    if budget is None:
        return "not_configured"
    if count is None:
        return "unknown"
    return "ok" if count <= budget else "over"


# F3.7 context-reduction preservation order (highest value first).
_PRESERVATION_TIERS: dict[str, int] = {
    "main_candidate": 0,
    "ticket_evidence": 1,
    "associated_test": 2,
    "related_class": 3,
    "dependency": 4,
}
_TIER_FALLBACK = 5


def _tier(source: str) -> int:
    return _PRESERVATION_TIERS.get(source, _TIER_FALLBACK)


def file_drop_order(items: list[Any]) -> list[str]:
    """File paths in drop order (lowest preservation value first).

    ``items`` are ContextItem-like objects exposing ``file_path``,
    ``included``, ``source`` and ``final_score``. Deterministic: tier, then
    ascending final score, then path for stability.
    """
    included = [item for item in items if getattr(item, "included", True)]
    ordered = sorted(
        included,
        key=lambda item: (
            -_tier(getattr(item, "source", "")),
            getattr(item, "final_score", 0.0),
            getattr(item, "file_path", ""),
        ),
    )
    paths: list[str] = []
    for item in ordered:
        path = getattr(item, "file_path", "") or ""
        if path and path not in paths:
            paths.append(path)
    return paths


def compute_costs(
    pricing: dict[str, Any],
    provider: str,
    model: str,
    input_tokens: int | None,
    output_tokens: int | None,
    max_output_tokens: int | None,
) -> dict[str, Any]:
    """Pre-call and post-call cost estimates from configured pricing.

    ``pricing`` is nested ``{provider: {model_id: {input_per_million,
    output_per_million}}}``. Returns keys ``estimated_max_cost`` and
    ``actual_operational_cost_estimate`` (floats or None). None when pricing
    is not configured for the provider/model — costs are then reported
    unavailable, never fabricated.
    """
    spec = (pricing.get(provider) or {}).get(model) or {}
    input_price = spec.get("input_per_million")
    output_price = spec.get("output_per_million")
    if not isinstance(input_price, (int, float)) or not isinstance(output_price, (int, float)):
        return {
            "estimated_max_cost": None,
            "actual_operational_cost_estimate": None,
        }
    input_tokens = int(input_tokens or 0)
    output_tokens = int(output_tokens or 0)
    max_output = int(max_output_tokens or 0)
    estimated_max_cost = round(
        input_tokens / 1_000_000 * input_price
        + max_output / 1_000_000 * output_price,
        6,
    )
    actual_operational_cost_estimate = round(
        input_tokens / 1_000_000 * input_price
        + output_tokens / 1_000_000 * output_price,
        6,
    )
    return {
        "estimated_max_cost": estimated_max_cost,
        "actual_operational_cost_estimate": actual_operational_cost_estimate,
    }
