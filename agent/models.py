from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, field_validator

from agent.context import ContextBundle
from agent.priority import normalize_priority

class UserStory(BaseModel):
    id: str
    title: str
    description: str
    acceptanceCriteria: list[str] = []
    priority: str = "P3"

    @field_validator("priority", mode="before")
    @classmethod
    def _normalize_priority(cls, v: object) -> str:
        return normalize_priority(v)

class RepoChunk(BaseModel):
    path: str
    chunk_index: int
    text: str

class IndexedChunk(BaseModel):
    path: str
    chunk_index: int
    text: str
    embedding: list[float] = []

class ExactEdit(BaseModel):
    before: str
    after: str

class ProposalFile(BaseModel):
    path: str
    operation: str
    content: Optional[str] = None
    edits: list[ExactEdit] = []

class PatchProposalPlan(BaseModel):
    storyId: str
    summary: str
    files: list[ProposalFile]
    tests: list[str] = []
    notes: list[str] = []

class PatchFile(BaseModel):
    path: str
    operation: str
    content: Optional[str] = None
    base_file_hash: Optional[str] = None

class PatchPlan(BaseModel):
    storyId: str
    summary: str
    files: list[PatchFile]
    tests: list[str] = []
    notes: list[str] = []

class AnalysisResult(BaseModel):
    success: bool
    findings: list[str]
    report: str

class TestRunResult(BaseModel):
    command: str
    exit_code: int
    output: str
    @property
    def failed(self) -> bool:
        return self.exit_code != 0

class TestFailureContext(BaseModel):
    story_id: str
    summary: str
    git_diff: str
    test_command: str
    test_output: str
    static_analysis_report: str
    attempt_number: int

class ProjectOverview(BaseModel):
    id: str
    name: str
    repo_root: str
    stories_file: str
    index_file: str
    test_command: str
    static_analysis_command: str

class PatchResult(BaseModel):
    project_id: str
    story_id: str
    plan: PatchPlan
    git_diff: str
    analysis: Optional[AnalysisResult] = None
    test_run: Optional[TestRunResult] = None
    attempts_used: int
    pending_review: bool = False
    # F2 hybrid contextual RAG: which retrieval mode produced the context
    # and the explainable context bundle (None when no store/services).
    retrieval_mode: str = "legacy"
    context: Optional[ContextBundle] = None
