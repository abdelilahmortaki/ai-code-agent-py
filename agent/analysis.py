from __future__ import annotations
import re

from agent.config import ProjectConfig
from agent.codebase import CodebaseService
from agent.models import AnalysisResult

_TODO = re.compile(r"\b(TODO|FIXME|XXX)\b", re.IGNORECASE)
_PRINT = re.compile(r"System\.out\.println|System\.err\.println|printStackTrace\s*\(")
_SECRET = re.compile(r"(?i)(api[_-]?key|secret|token|password|passwd)\s*[:=]\s*\S+")
_SWALLOWED = re.compile(r"catch\s*\(.*?Exception.*?\)\s*\{\s*\}", re.DOTALL)
_RETURN_NULL = re.compile(r"return\s+null\s*;")


class StaticAnalysisService:
    def __init__(self, codebase: CodebaseService) -> None:
        self.codebase = codebase

    def analyze(self, project: ProjectConfig) -> AnalysisResult:
        docs = self.codebase.scan(project)
        findings: list[str] = []

        for path, content in docs:
            if not content or content.isspace():
                continue
            if _TODO.search(content):
                findings.append(f"{path}: contains TODO/FIXME markers")
            if _PRINT.search(content):
                findings.append(f"{path}: contains console logging / stack traces")
            if _SECRET.search(content):
                findings.append(f"{path}: possible secret-like literal")
            if _SWALLOWED.search(content):
                findings.append(f"{path}: swallowed exception block detected")
            if _RETURN_NULL.search(content):
                findings.append(f"{path}: returns null explicitly")

        success = not findings
        report = (
            "Static analysis OK: no basic hygiene issues found."
            if success
            else "\n".join(findings)
        )
        return AnalysisResult(success=success, findings=findings, report=report)
