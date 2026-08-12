from __future__ import annotations
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel

# Load .env from the project root (ai-code-agent-py/.env) before anything else
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


class ProjectConfig(BaseModel):
    id: str
    name: str
    repo_root: str
    stories_file: str
    index_file: str
    test_command: str = "mvn test"
    static_analysis_command: str = ""
    allowed_write_extensions: list[str] = [
        "java", "xml", "yml", "yaml", "properties", "json", "md", "txt"
    ]
    excluded_directories: list[str] = [".git", "target", ".agent"]

    def normalized_allowed_write_extensions(self) -> list[str]:
        return [ext.lstrip(".").lower() for ext in self.allowed_write_extensions]

    def normalized_excluded_directories(self) -> list[str]:
        return [d.strip("/\\") for d in self.excluded_directories]


class AgentConfig(BaseModel):
    active_project: str = ""
    max_attempts: int = 3
    max_context_files: int = 6   # number of full files sent to OpenAI (not chunks)
    max_file_chars: int = 12000
    projects: list[ProjectConfig] = []

    def project_by_id(self, project_id: str) -> ProjectConfig:
        target_id = project_id or self.active_project
        if not target_id and self.projects:
            return self.projects[0]
        for p in self.projects:
            if p.id.lower() == target_id.lower():
                return p
        raise ValueError(f"Project not found: {project_id}")


class BedrockConfig(BaseModel):
    region: str = "us-east-1"
    model_id: str = "qwen.qwen3-coder-30b-a3b-v1:0"
    embeddings_model_id: str = "amazon.titan-embed-text-v2:0"
    temperature: float = 0.1
    max_tokens: int = 32384
    expected_account_id: str = ""


class OpenAiConfig(BaseModel):  # kept for backward-compat, unused when Bedrock is active
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    responses_model: str = "gpt-4o"
    embeddings_model: str = "text-embedding-3-small"


class Settings(BaseModel):
    bedrock: BedrockConfig = BedrockConfig()
    agent: AgentConfig = AgentConfig()

    @classmethod
    def load(cls, config_path: str = "config.yml") -> "Settings":
        data: dict = {}
        if Path(config_path).exists():
            with open(config_path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

        bd = data.get("bedrock", {})
        bedrock_cfg = BedrockConfig(
            region=os.getenv("AWS_REGION") or bd.get("region", "us-east-1"),
            model_id=os.getenv("BEDROCK_MODEL_ID") or bd.get("model_id", "qwen.qwen3-coder-30b-a3b-v1:0"),
            embeddings_model_id=bd.get("embeddings_model_id", "amazon.titan-embed-text-v2:0"),
            temperature=float(os.getenv("BEDROCK_TEMPERATURE") or bd.get("temperature", 0.1)),
            max_tokens=int(os.getenv("BEDROCK_MAX_TOKENS") or bd.get("max_tokens", 16384)),
            expected_account_id=os.getenv("AWS_EXPECTED_ACCOUNT_ID") or bd.get("expected_account_id", ""),
        )

        agent_data: dict = dict(data.get("agent", {}))
        projects_data: list = agent_data.pop("projects", [])
        agent_cfg = AgentConfig(
            **agent_data,
            projects=[ProjectConfig(**p) for p in projects_data],
        )

        return cls(bedrock=bedrock_cfg, agent=agent_cfg)
