from __future__ import annotations

import difflib
from typing import TYPE_CHECKING, Callable

from agent.config import AgentConfig
from agent.analysis import StaticAnalysisService
from agent.codebase import CodebaseService
from agent.git_service import GitDiffService
from agent.guard import PromptGuardService
from agent.index import SemanticIndexService
from agent.models import PatchPlan, PatchProposalPlan, PatchResult, ProjectOverview, TestFailureContext, UserStory
from agent.materializer import PatchMaterializer
from agent.guard import PromptGuardService
from agent.patch import PatchApplierService
from agent.providers import GenerationProvider
from agent.runner import TestRunner
from agent.stories import StoryFileReader

if TYPE_CHECKING:
    from agent.db.store import PgStore


_MAX_GENERATION_ATTEMPTS = 2
_LLM_INVOCATION_ERROR = "provider call failed"
_ACCEPT_FAILED_ERROR = "accept failed"
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
        prompt_guard: PromptGuardService | None = None,
        store: PgStore | None = None,
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
        self.materializer = PatchMaterializer()
        self.prompt_guard = prompt_guard or PromptGuardService()
        self.store = store
        # Pending plans awaiting user accept/reject — keyed by project_id.
        # Each entry is (plan, preview_diff, run_id) where run_id may be None
        # when run persistence is disabled.
        self._pending_plans: dict[str, tuple[PatchPlan, str, str | None]] = {}

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
        plan, preview_diff, run_id = entry
        project = self.config.project_by_id(project_id)
        try:
            self.patch_applier.apply(project, plan)
        except Exception:
            self._record_human_decision(run_id, "accepted", error=_ACCEPT_FAILED_ERROR)
            raise
        self._record_human_decision(run_id, "accepted")
        # Return the pre-computed diff — no git required
        return preview_diff

    def reject_plan(self, project_id: str) -> None:
        """Discard the pending patch plan for a project."""
        entry = self._pending_plans.pop(project_id, None)
        if entry is not None:
            self._record_human_decision(entry[2], "rejected")

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
        for pf in plan_files.files:
            current = self.codebase.read_file(project, pf.path) or ""
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
        run_id: str | None = None,
    ) -> tuple[PatchPlan, int]:
        feedback = ""
        for attempt in range(1, _MAX_GENERATION_ATTEMPTS + 1):
            log(f"Generation attempt {attempt}/{_MAX_GENERATION_ATTEMPTS}")
            try:
                proposal = self.generation_provider.generate_patch_plan(
                    project,
                    story,
                    relevant_files,
                    static_analysis,
                    log,
                    validation_feedback=feedback,
                )
            except Exception:
                self._record_llm_invocation(run_id, log, status="failed", error=_LLM_INVOCATION_ERROR)
                raise
            self._record_llm_invocation(run_id, log, status="ok")
            try:
                plan = self.materializer.materialize(project, proposal)
                self.prompt_guard.validate_materialized_minimality(project, plan, story)
            except ValueError as exc:
                if str(exc) not in {"UNRELATED_FORMATTING_CHANGED", "EDIT_ANCHOR_NOT_FOUND", "EDIT_ANCHOR_AMBIGUOUS"} or attempt == _MAX_GENERATION_ATTEMPTS:
                    raise
                log(f"Proposal rejected: {exc}")
                log("Regenerating with preservation feedback")
                feedback = _PRESERVATION_FEEDBACK + f"\nPrevious validation error: {exc}"
                continue
            log("Proposal validated")
            return plan, attempt
        raise RuntimeError("Generation attempts exhausted")

    def _run(self, project, story: UserStory, log: Callable[[str], None] = _noop) -> PatchResult:
        self._pending_plans.pop(project.id, None)
        run_id = self._start_run(project, story, log)
        try:
            return self._run_lifecycle(project, story, log, run_id)
        except Exception:
            self._fail_run(run_id, log)
            raise

    def _run_lifecycle(self, project, story: UserStory, log: Callable[[str], None], run_id: str | None) -> PatchResult:
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
            project, story, relevant_files, pre_analysis.report, log, run_id=run_id
        )
        ops = ", ".join(sorted({f.operation for f in plan.files})) or "none"
        log(f"Patch plan: {len(plan.files)} file(s) [{ops}] — {plan.summary}")

        log("Computing diff preview…")
        preview_diff = self._preview_diff(project, plan)
        log(f"Diff ready — {preview_diff.count(chr(10))} lines changed")

        # Store plan + pre-computed diff for accept/reject — do NOT write files yet
        self._pending_plans[project.id] = (plan, preview_diff, run_id)
        self._set_run_status(run_id, "awaiting_review", log)
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

    # ------------------------------------------------------------------ run persistence (best-effort)

    def _start_run(self, project, story: UserStory, log: Callable[[str], None]) -> str | None:
        """Create a persisted run record; returns its id or None when skipped."""
        if self.store is None:
            return None
        try:
            project_version_id = None
            db_project = self.store.find_project_by_external_id(project.id)
            if db_project is not None:
                version = self.store.latest_version(db_project["id"])
                if version is not None:
                    project_version_id = version["id"]
            run = self.store.create_run(
                project_id=project.id,
                project_version_id=project_version_id,
                story_snapshot=story.model_dump(),
                priority=story.priority,
            )
            log(f"Run recorded: {run['id']}")
            return run["id"]
        except Exception as exc:
            log(f"Run persistence skipped: {exc}")
            return None

    def _set_run_status(self, run_id: str | None, status: str, log: Callable[[str], None]) -> None:
        if run_id is None or self.store is None:
            return
        try:
            self.store.update_run_status(run_id, status)
        except Exception as exc:
            log(f"Run status update skipped: {exc}")

    def _fail_run(self, run_id: str | None, log: Callable[[str], None]) -> None:
        if run_id is None or self.store is None:
            return
        try:
            self.store.complete_run(run_id, "failed", error="generation failed")
        except Exception as exc:
            log(f"Run failure update skipped: {exc}")

    def _record_llm_invocation(
        self,
        run_id: str | None,
        log: Callable[[str], None],
        status: str,
        error: str | None = None,
    ) -> None:
        if run_id is None or self.store is None:
            return
        try:
            provider = getattr(self.generation_provider, "provider_name", None)
            model = getattr(self.generation_provider, "model_identity", None)
            self.store.insert_llm_invocation(
                run_id, provider=provider, model=model, status=status, error=error
            )
        except Exception as exc:
            log(f"Invocation persistence skipped: {exc}")

    def _record_human_decision(
        self, run_id: str | None, decision: str, error: str | None = None
    ) -> None:
        if run_id is None or self.store is None:
            return
        try:
            if error is not None:
                self.store.complete_run(run_id, "failed", human_decision=decision, error=error)
            else:
                self.store.complete_run(run_id, "completed", human_decision=decision)
        except Exception:
            pass
