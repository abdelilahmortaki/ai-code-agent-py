from __future__ import annotations
import json

import httpx

from agent.config import OpenAiConfig, ProjectConfig
from agent.guard import PromptGuardService
from agent.models import PatchFile, PatchPlan, TestFailureContext, UserStory

# JSON Schema for structured OpenAI output — mirrors PatchPlanSchemaFactory.java
_PATCH_PLAN_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "storyId":  {"type": "string", "description": "Story identifier"},
        "summary":  {"type": "string", "description": "Technical summary of changes"},
        "files": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "path":      {"type": "string", "description": "Relative file path"},
                    "operation": {"type": "string", "enum": ["create", "modify", "delete"]},
                    "content":   {
                        "anyOf": [{"type": "string"}, {"type": "null"}],
                        "description": "Complete new file content (null for delete)",
                    },
                },
                "required": ["path", "operation", "content"],
            },
        },
        "tests": {"type": "array", "items": {"type": "string"}},
        "notes": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["storyId", "summary", "files", "tests", "notes"],
}


class OpenAiService:
    def __init__(self, cfg: OpenAiConfig, prompt_guard: PromptGuardService) -> None:
        self.cfg = cfg
        self.prompt_guard = prompt_guard
        self._client = httpx.Client(
            base_url=cfg.base_url,
            headers={
                "Authorization": f"Bearer {cfg.api_key}",
                "Content-Type": "application/json",
            },
            timeout=180.0,
        )

    def generate_patch_plan(
        self,
        project: ProjectConfig,
        story: UserStory,
        relevant_files: list[tuple[str, str]],  # (path, full_content)
        static_analysis: str,
    ) -> PatchPlan:
        context = self.prompt_guard.build_prompt(project, story, relevant_files, static_analysis)
        plan = self._call_structured(context.prompt, "spring_boot_patch_plan")
        plan = self.prompt_guard.restore_plan(context, plan)
        return self.prompt_guard.validate_minimality(context, plan, story)

    def generate_fix_plan(self, ctx: TestFailureContext) -> PatchPlan:
        prompt = (
            "A previous patch failed.\n\n"
            f"storyId: {ctx.story_id}\n"
            f"summary: {ctx.summary}\n"
            f"attempt: {ctx.attempt_number}\n\n"
            f"GIT_DIFF:\n{self.prompt_guard.sanitize_for_fix(ctx.git_diff)}\n\n"
            f"TEST_COMMAND:\n{ctx.test_command}\n\n"
            f"TEST_OUTPUT:\n{self.prompt_guard.sanitize_for_fix(ctx.test_output)}\n\n"
            f"STATIC_ANALYSIS:\n{self.prompt_guard.sanitize_for_fix(ctx.static_analysis_report)}\n\n"
            "Return the minimal correction as JSON only."
        )
        return self._call_structured(prompt, "spring_boot_fix_plan")

    # ------------------------------------------------------------------ private

    def _call_structured(self, prompt: str, schema_name: str) -> PatchPlan:
        payload = {
            "model": self.cfg.responses_model,
            "reasoning": {"effort": "medium"},
            "instructions": (
                "You are a senior Java/Spring Boot engineer. "
                "Produce minimal, coherent, testable changes."
            ),
            "max_output_tokens": 8000,
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt}],
                }
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "strict": True,
                    "schema": _PATCH_PLAN_SCHEMA,
                }
            },
        }
        resp = self._client.post("/responses", json=payload)
        resp.raise_for_status()
        raw_json = self._extract_text(resp.json())
        data = json.loads(raw_json)
        return PatchPlan(
            storyId=data["storyId"],
            summary=data["summary"],
            files=[PatchFile(**f) for f in data.get("files", [])],
            tests=data.get("tests", []),
            notes=data.get("notes", []),
        )

    @staticmethod
    def _extract_text(response: dict) -> str:
        direct = response.get("output_text")
        if direct:
            return direct
        parts: list[str] = []
        for item in response.get("output", []):
            for c in item.get("content", []):
                if c.get("type") in ("output_text", "text"):
                    parts.append(c.get("text", ""))
        result = "".join(parts)
        if not result:
            raise RuntimeError("No usable text in OpenAI response")
        return result
