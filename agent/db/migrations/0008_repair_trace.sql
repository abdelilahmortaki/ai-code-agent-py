-- F5 repair trace identity. One row per bounded repair attempt.
ALTER TABLE llm_invocations ADD COLUMN IF NOT EXISTS attempt_number INTEGER;
ALTER TABLE llm_invocations ADD COLUMN IF NOT EXISTS context_indicator TEXT;
