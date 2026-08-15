"""P1-P4 smart routing (F4.2-F4.5).

Deterministic, provider-neutral routing that runs AFTER provider selection
(AI_PROVIDER). Routing never switches Azure<->Bedrock.

- complexity: deterministic LOW/MEDIUM/HIGH scoring from current evidence
  (candidate modules, relevant symbol count, graph breadth, API impact,
  DB impact, stack-trace presence) — never user-editable, never an LLM.
- policy: (priority, complexity) -> model profile (strong / economical);
  a profile resolves to provider-local retrieval knobs, context budget and
  generation options.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

COMPLEXITY_LOW = "LOW"
COMPLEXITY_MEDIUM = "MEDIUM"
COMPLEXITY_HIGH = "HIGH"

_API_HINTS = re.compile(
    r"\b(controller|endpoint|rest|api|http|get|post|put|delete|status\s*code|"
    r"response|request|dto)\b",
    re.IGNORECASE,
)
_DB_HINTS = re.compile(
    r"\b(repository|entity|persistence|jpa|database|sql|schema|migration|"
    r"spring.?data|h2|query)\b",
    re.IGNORECASE,
)
_STACK_HINTS = re.compile(r"\b(at\s+[a-z_$][\w.$]*\([^)]*\)|exception|stacktrace)\b", re.IGNORECASE)


def compute_complexity(
    story_text: str,
    retrieval: dict[str, Any] | None,
    selected_files: list[str],
) -> tuple[str, int]:
    """Deterministic complexity label + numeric tier.

    Scoring policy (documented, same inputs -> same output):
    - relevant_symbols: number of ranked retrieval items;
    - graph_breadth: graph neighbor evidence count;
    - modules: distinct top-level module names among selected files;
    - api_impact: story mentions controllers/endpoints/API/dto...
    - db_impact: story mentions repository/entity/persistence...
    - stack_trace: story contains stack-trace-like lines.

    tier = min(3, 1
      + (1 if relevant_symbols >= 8 else 0)
      + (1 if graph_breadth >= 4 or modules >= 3 else 0)
      + (1 if api_impact or db_impact else 0)
      + (1 if stack_trace else 0))
    HIGH: tier >= 3, MEDIUM: tier == 2, LOW: tier <= 1.
    """
    items = retrieval.get("items", []) if retrieval else []
    counts = (retrieval or {}).get("counts") or {}
    relevant_symbols = len(items)
    graph_breadth = int(counts.get("graph_neighbors", 0) or 0)

    modules: set[str] = set()
    for path in selected_files or []:
        parts = path.split("/")
        if parts and parts[0].startswith("shoppoc-"):
            modules.add(parts[0])
    modules_count = len(modules)

    api_impact = bool(_API_HINTS.search(story_text))
    db_impact = bool(_DB_HINTS.search(story_text))
    stack_trace = bool(_STACK_HINTS.search(story_text))

    tier = 1
    if relevant_symbols >= 8:
        tier += 1
    if graph_breadth >= 4 or modules_count >= 3:
        tier += 1
    if api_impact or db_impact:
        tier += 1
    if stack_trace:
        tier += 1
    tier = min(3, tier)

    label = COMPLEXITY_HIGH if tier >= 3 else (COMPLEXITY_MEDIUM if tier == 2 else COMPLEXITY_LOW)
    return label, tier


@dataclass(frozen=True)
class RouteDecision:
    priority: str
    complexity_label: str
    profile: str
    graph_depth: int
    top_k: int
    max_related: int
    context_budget: int | None
    max_context_files: int
    max_output_tokens: int | None
    model_or_deployment: str | None
    reason: str


@dataclass
class RoutingConfig:
    profiles: dict[str, dict[str, Any]] = field(default_factory=dict)
    policy: dict[str, dict[str, str]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "RoutingConfig":
        data = data or {}
        return cls(
            profiles=dict(data.get("profiles") or {}),
            policy=dict(data.get("policy") or {}),
        )


def select_route(
    priority: str,
    complexity_label: str,
    routing: RoutingConfig,
    default_profile: str = "strong",
) -> tuple[str, dict[str, Any]]:
    """Resolve the profile name + profile settings for (priority, complexity)."""
    complexity_tier = routing.policy.get(priority) or {}
    profile = complexity_tier.get(complexity_label) or default_profile
    settings = routing.profiles.get(profile) or {}
    return profile, settings


def build_route_decision(
    priority: str,
    complexity_label: str,
    routing: RoutingConfig,
    provider_name: str,
    configured_model: str,
    model_overrides: dict[str, Any] | None = None,
    default_profile: str = "strong",
) -> RouteDecision:
    """Build the full deterministic route decision for a request."""
    profile, settings = select_route(priority, complexity_label, routing, default_profile)
    model_overrides = model_overrides or {}
    profile_override = (model_overrides.get(provider_name) or {}).get(profile)
    model_or_deployment = profile_override or configured_model
    graph_depth = int(settings.get("graph_depth", 1))
    top_k = int(settings.get("top_k", 10))
    max_related = int(settings.get("max_related", 8))
    context_budget = settings.get("context_budget")
    context_budget = int(context_budget) if context_budget is not None else None
    max_context_files = int(settings.get("max_context_files", 6))
    max_output = settings.get("max_output_tokens")
    max_output = int(max_output) if max_output is not None else None
    reason = f"{priority} + {complexity_label} -> {profile} profile"
    return RouteDecision(
        priority=priority,
        complexity_label=complexity_label,
        profile=profile,
        graph_depth=graph_depth,
        top_k=top_k,
        max_related=max_related,
        context_budget=context_budget,
        max_context_files=max_context_files,
        max_output_tokens=max_output,
        model_or_deployment=model_or_deployment,
        reason=reason,
    )
