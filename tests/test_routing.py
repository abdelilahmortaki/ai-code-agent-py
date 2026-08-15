"""F4 smart routing contract tests (#32-#36) — pure functions."""

from __future__ import annotations

from agent.routing import (
    RoutingConfig,
    build_route_decision,
    compute_complexity,
)

_POLICY = {
    "profiles": {
        "strong": {
            "graph_depth": 1, "top_k": 10, "max_related": 8,
            "context_budget": None, "max_context_files": 6,
            "max_output_tokens": 16384,
        },
        "economical": {
            "graph_depth": 0, "top_k": 5, "max_related": 4,
            "context_budget": 6000, "max_context_files": 3,
            "max_output_tokens": 8192,
        },
    },
    "policy": {
        "P1": {"LOW": "strong", "MEDIUM": "strong", "HIGH": "strong"},
        "P3": {"LOW": "economical", "MEDIUM": "economical", "HIGH": "strong"},
        "P4": {"LOW": "economical", "MEDIUM": "economical", "HIGH": "economical"},
    },
}


def _config():
    return RoutingConfig(
        profiles=_POLICY["profiles"],
        policy=_POLICY["policy"],
    )


def test_complexity_deterministic_low():
    story = "Validate product name length on create"
    retrieval = {"items": [{}], "counts": {"graph_neighbors": 0}}
    files = ["shoppoc-catalog/src/main/java/com/shoppoc/catalog/domain/Product.java"]
    assert compute_complexity(story, retrieval, files) == ("LOW", 1)


def test_complexity_deterministic_high():
    story = (
        "Restrict order quantity with pessimistic lock at GET /api/v1/orders "
        "in OrderController; the JPA entity and repository must change and "
        "the stack trace at com.shoppoc...Exception must be handled"
    )
    retrieval = {"items": list(range(12)), "counts": {"graph_neighbors": 6}}
    files = [
        "shoppoc-order/src/main/java/com/shoppoc/order/application/OrderApplicationService.java",
        "shoppoc-catalog/src/main/java/com/shoppoc/catalog/domain/Product.java",
        "shoppoc-shared/src/main/java/com/shoppoc/shared/money/Money.java",
    ]
    assert compute_complexity(story, retrieval, files) == ("HIGH", 3)


def test_complexity_never_user_editable():
    assert not hasattr(compute_complexity, "settings")
    assert compute_complexity("x", {"items": []}, []) == ("LOW", 1)


def test_acceptance_a_priority_changes_route_only():
    """Same ticket/evidence/complexity; only priority changes."""
    overrides = {"azure": {"strong": "dep-strong", "economical": "dep-econ"}}
    base = dict(complexity_label="LOW", routing=_config(), provider_name="azure",
                configured_model="default-dep", model_overrides=overrides)
    p1 = build_route_decision("P1", **base)
    p4 = build_route_decision("P4", **base)
    assert p1.profile == "strong"
    assert p4.profile == "economical"
    assert p1.model_or_deployment == "dep-strong"
    assert p4.model_or_deployment == "dep-econ"
    assert p1.graph_depth != p4.graph_depth
    assert p1.reason != p4.reason


def test_acceptance_b_complexity_changes_route_only():
    """Fixed priority; only complexity changes."""
    overrides = {"azure": {"strong": "dep-strong", "economical": "dep-econ"}}
    base = dict(priority="P3", routing=_config(), provider_name="azure",
                configured_model="default-dep", model_overrides=overrides)
    low = build_route_decision(complexity_label="LOW", **base)
    high = build_route_decision(complexity_label="HIGH", **base)
    assert low.profile == "economical"
    assert high.profile == "strong"
    assert low.max_output_tokens != high.max_output_tokens


def test_route_reason_is_deterministic():
    a = build_route_decision("P1", "HIGH", _config(), "azure", "dep", {})
    b = build_route_decision("P1", "HIGH", _config(), "azure", "dep", {})
    assert a.reason == b.reason == "P1 + HIGH -> strong profile"
