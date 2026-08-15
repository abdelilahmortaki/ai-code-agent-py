# F4 Smart Routing Acceptance (#32-#36) — real VM + real Azure

Migration 0007 applied: runs.complexity_label TEXT + runs.routing JSONB
(integer complexity column preserved for numeric tiers; strings never written
into it).

## #32 Complexity — deterministic, not user-editable
- compute_complexity: relevant_symbols, graph_breadth, modules, api_impact,
  db_impact, stack_trace -> tier 1..3 -> LOW/MEDIUM/HIGH. Same inputs -> same
  output. Real run: HIGH (tier 3) for a controller+repository+stack ticket.
- Contract tests: LOW case -> ('LOW',1); HIGH case -> ('HIGH',3).

## #33 Routing policy — config-driven, provider-local
- Profiles strong/economical in config.yml (graph_depth, top_k, max_related,
  context_budget, max_context_files, max_output_tokens).
- Policy (priority, complexity) -> profile. model_overrides map profile ->
  provider-local deployment/model id; unset -> configured default (VM's one
  real Azure deployment). Routing after AI_PROVIDER selection; never switches
  providers.

## #34 End-to-end applied route
- ExecutionOptions (model_or_deployment, max_output_tokens) on provider
  protocol + Azure + Bedrock adapters; Azure responses.create honors
  max_output_tokens.
- Orchestrator: pass-1 retrieval -> complexity -> route decision -> pass-2
  retrieval with routed graph_depth/top_k -> context with routed budget /
  max_context_files -> generation with routed model/max_tokens.
- F3 counting uses the routed identity (max_output_tokens from route).

## #35 Routing reason persisted
- Real runs: "P1 + HIGH -> strong profile", "P4 + HIGH -> economical profile".
- Persisted on runs.routing.reason AND llm_invocations.routing_reason.

## Acceptance A (priority effect) — REAL runs, same ticket/evidence/complexity
- P1 -> strong profile, graph depth 1, max_output_tokens 16384
- P4 -> economical profile, graph depth 0, max_output_tokens 8192

## Acceptance B (complexity effect) — contract tests
- Fixed P3: LOW -> economical; HIGH -> strong.

## #36 Routing UI
- FinOps tab now shows a Routing section: priority, complexity, profile,
  graph depth, context budget, max output tokens, model/deployment, reason.
- Runs API exposes complexity_label + routing. Screenshot unavailable (no
  browser on VM) — documented.
- 46 tests pass (40 + 6 routing).
