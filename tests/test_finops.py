"""Focused FinOps contract tests (F3.3-F3.7) — pure functions only."""

from __future__ import annotations

from agent.finops import (
    compute_costs,
    compute_effective_budget,
    count_prompt,
    estimate_tokens,
    evaluate,
    file_drop_order,
)


def test_estimate_tokens_chars_per_4():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcdefgh") == 2


class _CountingProvider:
    provider_name = "fake"

    def count_tokens(self, text):  # noqa: D401
        return 7


class _NonCountingProvider:
    provider_name = "fake"

    def count_tokens(self, text):  # noqa: D401
        return None


def test_count_prompt_exact_vs_estimate():
    exact, method = count_prompt(_CountingProvider(), "hello world")
    assert exact == 7 and method == "exact:provider"
    est, method = count_prompt(_NonCountingProvider(), "abcdefgh")
    assert est == 2 and method == "estimate:chars/4"


def test_effective_budget_is_min_of_configured_limits():
    assert compute_effective_budget([None, None]) is None
    assert compute_effective_budget([1000, None, 500]) == 500
    assert compute_effective_budget([0, 1000]) == 1000


def test_evaluate_threshold():
    assert evaluate(100, 1000) == "ok"
    assert evaluate(1000, 500) == "over"
    assert evaluate(100, None) == "not_configured"


def test_costs_use_configured_pricing_and_report_unavailable_otherwise():
    pricing = {"azure": {"gpt-5-mini": {"input_per_million": 1.10, "output_per_million": 4.40}}}
    costs = compute_costs(pricing, "azure", "gpt-5-mini", 10_000, 2_000, 8_192)
    assert costs["estimated_max_cost"] is not None
    assert costs["actual_operational_cost_estimate"] is not None
    # 10000 in * 1.10 + 2000 out * 4.40 per million
    assert abs(costs["actual_operational_cost_estimate"] - (0.011 + 0.0088)) < 1e-6
    unavailable = compute_costs(pricing, "azure", "unknown-model", 100, 100, 100)
    assert unavailable["estimated_max_cost"] is None
    assert unavailable["actual_operational_cost_estimate"] is None


class _Item:
    def __init__(self, path, source, score):
        self.file_path = path
        self.source = source
        self.final_score = score
        self.included = True


def test_drop_order_preserves_high_value_evidence_last():
    items = [
        _Item("main.java", "main_candidate", 0.9),
        _Item("dep.java", "dependency", 0.5),
        _Item("ticket.java", "ticket_evidence", 0.7),
    ]
    assert file_drop_order(items) == ["dep.java", "ticket.java", "main.java"]
