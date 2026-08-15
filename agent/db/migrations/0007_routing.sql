-- F4 Smart routing: complexity label + applied route on runs.
-- Public F4 complexity semantics are LOW/MEDIUM/HIGH (text). The legacy
-- integer column stays for numeric tiers; a dedicated TEXT column carries
-- the label so strings are never written into the integer column.

ALTER TABLE runs ADD COLUMN IF NOT EXISTS complexity_label TEXT;
ALTER TABLE runs ADD COLUMN IF NOT EXISTS routing JSONB;
