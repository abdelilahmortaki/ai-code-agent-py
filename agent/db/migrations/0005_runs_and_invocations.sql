-- F3.1 Minimal run and LLM invocation persistence.

-- One row per execution (process_story / process_adhoc_story funnel through
-- orchestrator._run, which creates exactly one run per call).
-- project_id stores the runtime project id (projects.external_id); the PG
-- project_versions link is optional and may be NULL when no version exists.
-- complexity stays NULL until F4 lands.
CREATE TABLE IF NOT EXISTS runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id TEXT NOT NULL,
    project_version_id UUID REFERENCES project_versions(id) ON DELETE SET NULL,
    story_snapshot JSONB NOT NULL,
    priority TEXT NOT NULL,
    complexity INTEGER,
    status TEXT NOT NULL,
    test_status TEXT,
    human_decision TEXT,
    error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ
);

-- Invocation skeleton linked to its run; full consumption fields land in
-- F3.6/F3.7.
CREATE TABLE IF NOT EXISTS llm_invocations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id UUID NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    provider TEXT,
    model TEXT,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    total_tokens INTEGER,
    status TEXT,
    error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_llm_invocations_run_id ON llm_invocations(run_id);
CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(created_at);
