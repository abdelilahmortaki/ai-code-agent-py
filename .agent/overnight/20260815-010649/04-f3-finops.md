# F3 FinOps Acceptance (#25-#30) — real VM + real Azure

Migration 0006 applied to live DB (llm_invocations extended: count_method,
estimated_input_tokens, effective_input_budget, threshold_result,
reduction_outcome, max_output_tokens, estimated_max_cost,
actual_operational_cost_estimate, routing_reason, selected_context, prompt_hash).

## Real Azure generation — FULL TRACE (default config, no limits)
- count_method: estimate:chars/4 (Azure has no bundled tokenizer; documented fallback)
- estimated_input_tokens: 2582; threshold_result: not_configured; reduction: no_limits
- routing_reason: default configured provider/model; prompt_hash: 7eaaaf7d07a9
- ACTUAL Azure usage persisted: prompt_tokens=2232, completion_tokens=605, total=2837

## OVER/reduction acceptance (ticket_limit=1500) — REAL generation
- estimated_input_tokens 1071 <= budget 1500, threshold_result ok after reduction
- reduction_outcome: dropped CatalogApplicationService, NotFoundException,
  GlobalExceptionHandlerTest (preservation order: dependencies/tests before main candidate)
- estimated_max_cost 0.001178 = actual_operational_cost_estimate 0.001178
- actual usage: 1065 in / 611 out

## STOP acceptance (ticket_limit=300) — NO Azure call
- HTTP 429 FINOPS_STOP; persisted invocation status=stopped, threshold_result=over,
  estimated_input_tokens=1070, effective_input_budget=300, full reduction_outcome.
- No usage rows => no provider call made.

## Costs
- pricing configured as TEST values in config.yml (labeled, not authoritative);
  unconfigured models report cost unavailable (never fabricated).

## UI (F3.8)
- New FinOps tab: count method, estimated tokens, budget, threshold result,
  reduction outcome, max output tokens, estimated max cost, actual usage,
  actual operational cost, routing reason, prompt hash. Diff / Why this
  context? / Accept / Reject untouched.
- Browser/headless Chromium NOT available on VM -> screenshot not captured
  (documented, not faked). API + static wiring proven via /api/agent/runs.
