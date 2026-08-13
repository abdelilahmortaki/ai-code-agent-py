from __future__ import annotations
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, SecretStr

# Load .env from the project root (ai-code-agent-py/.env) before anything else
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

SUPPORTED_EXTENSIONS = {
    ".java", ".xml", ".yml", ".yaml", ".properties", ".json", ".md", ".txt"
}


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
    upload_allowed_extensions: list[str] = [ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS]
    upload_max_file_bytes: int = 153600
    upload_max_total_bytes: int = 10485760
    projects: list[ProjectConfig] = []

    def project_by_id(self, project_id: str) -> ProjectConfig:
        target_id = project_id or self.active_project
        if not target_id and self.projects:
            return self.projects[0]
        for p in self.projects:
            if p.id.lower() == target_id.lower():
                return p
        raise ValueError(f"Project not found: {project_id}")

    def normalized_upload_allowed_extensions(self) -> set[str]:
        return {f".{ext.lstrip('.').lower()}" for ext in self.upload_allowed_extensions}


class BedrockConfig(BaseModel):
    region: str = "us-east-1"
    model_id: str = "qwen.qwen3-coder-30b-a3b-v1:0"
    embeddings_model_id: str = "amazon.titan-embed-text-v2:0"
    temperature: float = 0.1
    max_tokens: int = 32384
    expected_account_id: str = ""


class AzureConfig(BaseModel):
    endpoint: str = ""
    api_key: SecretStr = SecretStr("")
    deployment: str = ""
    embedding_deployment: str = ""


class OpenAiConfig(BaseModel):  # kept for backward-compat, unused when Bedrock is active
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    responses_model: str = "gpt-4o"
    embeddings_model: str = "text-embedding-3-small"


class DatabaseConfig(BaseModel):
    url: SecretStr = SecretStr("")


class Settings(BaseModel):
    ai_provider: str = "bedrock"
    bedrock: BedrockConfig = BedrockConfig()
    azure: AzureConfig = AzureConfig()
    agent: AgentConfig = AgentConfig()
    database: DatabaseConfig = DatabaseConfig()

    @classmethod
    def load(cls, config_path: str = "config.yml") -> "Settings":
        data: dict = {}
        if Path(config_path).exists():
            with open(config_path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

        provider = (os.getenv("AI_PROVIDER") or "bedrock").strip().lower()
        if provider not in {"azure", "bedrock"}:
            raise ValueError("AI_PROVIDER must be one of: azure, bedrock")

        azure_cfg = AzureConfig()
        if provider == "azure":
            required = {
                "AZURE_OPENAI_ENDPOINT": os.getenv("AZURE_OPENAI_ENDPOINT"),
                "AZURE_OPENAI_API_KEY": os.getenv("AZURE_OPENAI_API_KEY"),
                "AZURE_OPENAI_DEPLOYMENT": os.getenv("AZURE_OPENAI_DEPLOYMENT"),
                "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
            }
            missing = [name for name, value in required.items() if not value or not value.strip()]
            if missing:
                raise ValueError(f"Missing required Azure configuration: {', '.join(missing)}")
            azure_cfg = AzureConfig(
                endpoint=required["AZURE_OPENAI_ENDPOINT"].strip(),
                api_key=SecretStr(required["AZURE_OPENAI_API_KEY"]),
                deployment=required["AZURE_OPENAI_DEPLOYMENT"].strip(),
                embedding_deployment=required["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"].strip(),
            )

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

        db_data: dict = dict(data.get("database", {}))
        db_url = os.getenv("DATABASE_URL") or str(db_data.get("url", "") or "")
        database_cfg = DatabaseConfig(url=SecretStr(db_url))

        return cls(
            ai_provider=provider,
            bedrock=bedrock_cfg,
            azure=azure_cfg,
            agent=agent_cfg,
            database=database_cfg,
        )
