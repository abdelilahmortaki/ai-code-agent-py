from __future__ import annotations

import difflib
from typing import Callable

from agent.config import AgentConfig
from agent.analysis import StaticAnalysisService
from agent.codebase import CodebaseService
from agent.git_service import GitDiffService
from agent.guard import PromptGuardService
from agent.index import SemanticIndexService
from agent.models import PatchPlan, PatchResult, ProjectOverview, TestFailureContext, UserStory
from agent.patch import PatchApplierService
from agent.providers import GenerationProvider
from agent.runner import TestRunner
from agent.stories import StoryFileReader


_MAX_GENERATION_ATTEMPTS = 2
_PRESERVATION_FEEDBACK = (
    "Previous proposal was rejected because it changed unrelated formatting. "
    "Regenerate from the authoritative source. "
    "Preserve all unrelated content exactly. "
    "Make only the requested semantic change. "
    "Do not change blank lines, indentation, trailing whitespace, EOF newline, "
    "comments, or ordering unless explicitly requested."
)


def _noop(msg: str) -> None:
    """Default no-op log callback."""
    pass


class AgentOrchestrator:
    def __init__(
        self,
        config: AgentConfig,
        codebase: CodebaseService,
        semantic_index: SemanticIndexService,
        generation_provider: GenerationProvider,
        patch_applier: PatchApplierService,
        git_diff: GitDiffService,
        static_analysis: StaticAnalysisService,
        test_runner: TestRunner,
        story_reader: StoryFileReader,
    ) -> None:
        self.config = config
        self.codebase = codebase
        self.semantic_index = semantic_index
        self.generation_provider = generation_provider
        self.patch_applier = patch_applier
        self.git_diff = git_diff
        self.static_analysis = static_analysis
        self.test_runner = test_runner
        self.story_reader = story_reader
        # Pending plans awaiting user accept/reject — keyed by project_id
        self._pending_plans: dict[str, object] = {}

    # ------------------------------------------------------------------ public

    def list_projects(self) -> list[ProjectOverview]:
        return [
            ProjectOverview(
                id=p.id,
                name=p.name,
                repo_root=p.repo_root,
                stories_file=p.stories_file,
                index_file=p.index_file,
                test_command=p.test_command,
                static_analysis_command=p.static_analysis_command,
            )
            for p in self.config.projects
        ]

    def rebuild_index(self, project_id: str) -> str:
        project = self.config.project_by_id(project_id)
        chunk_ids = self.semantic_index.rebuild(project)
        return f"Index rebuilt for '{project.id}' with {len(chunk_ids)} chunks"

    def accept_plan(self, project_id: str) -> str:
        """Apply the pending patch plan for a project after user approval."""
        from agent.models import PatchPlan
        entry = self._pending_plans.pop(project_id, None)
        if entry is None:
            raise ValueError(f"No pending plan for project '{project_id}'")
        plan, preview_diff = entry
        project = self.config.project_by_id(project_id)
        self.patch_applier.apply(project, plan.files)
        # Return the pre-computed diff — no git required
        return preview_diff

    def reject_plan(self, project_id: str) -> None:
        """Discard the pending patch plan for a project."""
        self._pending_plans.pop(project_id, None)

    def process_story(self, project_id: str, story_id: str, log_callback: Callable[[str], None] = _noop) -> PatchResult:
        project = self.config.project_by_id(project_id)
        stories = self.story_reader.read(project)
        story = next(
            (s for s in stories if s.id and s.id.lower() == story_id.lower()), None
        )
        if story is None:
            raise ValueError(f"Story not found: {story_id}")
        return self._run(project, story, log_callback)

    def process_adhoc_story(self, project_id: str, story: UserStory, log_callback: Callable[[str], None] = _noop) -> PatchResult:
        """Run the agent with a story supplied directly (not from the JSON file)."""
        project = self.config.project_by_id(project_id)
        return self._run(project, story, log_callback)

    # ------------------------------------------------------------------ private

    def _preview_diff(self, project, plan_files) -> str:
        """Compute a unified diff between current and proposed file contents
        without touching the filesystem."""
        output: list[str] = []
        for pf in plan_files:
            current = self.codebase.read_file(project, pf.path) or ""
            if pf.operation == "delete":
                new_content = ""
            else:
                new_content = pf.content or ""
            old_lines = current.splitlines(keepends=True)
            new_lines = new_content.splitlines(keepends=True)
            diff = difflib.unified_diff(
                old_lines, new_lines,
                fromfile=f"a/{pf.path}",
                tofile=f"b/{pf.path}",
            )
            chunk = list(diff)
            if chunk:
                output.append(f"diff --git a/{pf.path} b/{pf.path}\n")
                output.extend(chunk)
        return "".join(output)

    def _generate_patch_plan(
        self,
        project,
        story: UserStory,
        relevant_files: list[tuple[str, str]],
        static_analysis: str,
        log: Callable[[str], None],
    ) -> tuple[PatchPlan, int]:
        feedback = ""
        for attempt in range(1, _MAX_GENERATION_ATTEMPTS + 1):
            log(f"Generation attempt {attempt}/{_MAX_GENERATION_ATTEMPTS}")
            try:
                plan = self.generation_provider.generate_patch_plan(
                    project,
                    story,
                    relevant_files,
                    static_analysis,
                    log,
                    validation_feedback=feedback,
                )
            except ValueError as exc:
                if str(exc) != "UNRELATED_FORMATTING_CHANGED" or attempt == _MAX_GENERATION_ATTEMPTS:
                    raise
                log("Proposal rejected: unrelated formatting changed")
                log("Regenerating with preservation feedback")
                feedback = _PRESERVATION_FEEDBACK
                continue
            log("Proposal validated")
            return plan, attempt
        raise RuntimeError("Generation attempts exhausted")

    def _run(self, project, story: UserStory, log: Callable[[str], None] = _noop) -> PatchResult:
        self._pending_plans.pop(project.id, None)
        log("Checking semantic index…")
        if not self.semantic_index.exists(project):
            log("Index missing — building from codebase…")
            chunks = self.semantic_index.rebuild(project)
            log(f"Index built: {len(chunks)} chunks embedded")

        log("Running static pre-analysis…")
        pre_analysis = self.static_analysis.analyze(project)
        if pre_analysis.findings:
            log(f"Pre-analysis: {len(pre_analysis.findings)} hygiene issue(s) detected")

        log("Searching semantic index for relevant files…")
        relevant_paths = self.semantic_index.search(
            project,
            f"{story.title}\n{story.description}",
            self.config.max_context_files,
        )
        relevant_files = [
            (path, content)
            for path in relevant_paths
            if (content := self.codebase.read_file(project, path)) is not None
        ]
        total_chars = sum(len(c) for _, c in relevant_files)
        log(f"Context: {len(relevant_files)} file(s) loaded ({total_chars:,} chars total)")

        plan, attempts_used = self._generate_patch_plan(
            project, story, relevant_files, pre_analysis.report, log
        )
        ops = ", ".join(sorted({f.operation for f in plan.files})) or "none"
        log(f"Patch plan: {len(plan.files)} file(s) [{ops}] — {plan.summary}")

        log("Computing diff preview…")
        preview_diff = self._preview_diff(project, plan.files)
        log(f"Diff ready — {preview_diff.count(chr(10))} lines changed")

        # Store plan + pre-computed diff for accept/reject — do NOT write files yet
        self._pending_plans[project.id] = (plan, preview_diff)
        log("Awaiting your review — Accept or Reject the changes")

        return PatchResult(
            project_id=project.id,
            story_id=story.id,
            plan=plan,
            git_diff=preview_diff,
            analysis=pre_analysis,
            test_run=None,
            attempts_used=attempts_used,
            pending_review=True,
        )
