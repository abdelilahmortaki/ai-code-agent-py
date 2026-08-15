-- F3.7 Full consumption trace on llm_invocations.
-- One row per LLM invocation, carrying the complete FinOps trace:
-- count method, estimated input tokens, effective input budget,
-- threshold result, reduction outcome, max output tokens, pre-call
-- estimated max cost, post-call actual operational cost estimate,
-- routing reason, sanitized selected context and prompt hash.
-- All columns nullable so historical rows remain valid.

ALTER TABLE llm_invocations ADD COLUMN IF NOT EXISTS count_method TEXT;
ALTER TABLE llm_invocations ADD COLUMN IF NOT EXISTS estimated_input_tokens INTEGER;
ALTER TABLE llm_invocations ADD COLUMN IF NOT EXISTS effective_input_budget INTEGER;
ALTER TABLE llm_invocations ADD COLUMN IF NOT EXISTS threshold_result TEXT;
ALTER TABLE llm_invocations ADD COLUMN IF NOT EXISTS reduction_outcome TEXT;
ALTER TABLE llm_invocations ADD COLUMN IF NOT EXISTS max_output_tokens INTEGER;
ALTER TABLE llm_invocations ADD COLUMN IF NOT EXISTS estimated_max_cost NUMERIC;
ALTER TABLE llm_invocations ADD COLUMN IF NOT EXISTS actual_operational_cost_estimate NUMERIC;
ALTER TABLE llm_invocations ADD COLUMN IF NOT EXISTS routing_reason TEXT;
ALTER TABLE llm_invocations ADD COLUMN IF NOT EXISTS selected_context JSONB;
ALTER TABLE llm_invocations ADD COLUMN IF NOT EXISTS prompt_hash TEXT;
