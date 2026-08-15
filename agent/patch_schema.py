PATCH_PLAN_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "storyId": {"type": "string"},
        "summary": {"type": "string"},
        "files": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "path": {"type": "string"},
                    "operation": {"type": "string", "enum": ["create", "modify", "delete"]},
                    "content": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                    "reason": {"type": "string"},
                    "edits": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {"before": {"type": "string"}, "after": {"type": "string"}},
                            "required": ["before", "after"],
                        },
                    },
                },
                "required": ["path", "operation", "content", "reason", "edits"],
            },
        },
        "tests": {"type": "array", "items": {"type": "string"}},
        "notes": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["storyId", "summary", "files", "tests", "notes"],
}

PATCH_SCHEMA_HINT = """
For create, content is the complete new file and edits is empty.
For modify, content is null and edits contains exact before/after spans from the current file.
Each modify before span must be unique in the current file. Never use full-file content for modify.
For delete, content is null and edits is empty. Preserve protected values and unrelated content exactly.
reason is a short explanation of why this file is changed by this story.
"""