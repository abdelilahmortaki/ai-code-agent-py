from __future__ import annotations

import difflib
import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from agent.config import AgentConfig
from agent.analysis import StaticAnalysisService
from agent.codebase import CodebaseService
from agent.context import build_legacy_bundle
from agent.finops import (
    FinOpsStoppedError,
    compute_costs,
    compute_effective_budget,
    count_prompt,
    evaluate,
    file_drop_order,
)
from agent.git_service import GitDiffService
from agent.guard import PromptGuardService
from agent.index import SemanticIndexService
from agent.models import (
    ContextBundle,
    PatchPlan,
    PatchProposalPlan,
    PatchResult,
    ProjectOverview,
    TestFailureContext,
    UserStory,
)
from agent.materializer import PatchMaterializer
from agent.paths import resolve_within
from agent.patch import PatchApplierService
from agent.providers import GenerationProvider
from agent.runner import TestRunner
from agent.stories import StoryFileReader

if TYPE_CHECKING:
    from agent.context import ContextBuilder
    from agent.db.store import PgStore
    from agent.retrieval import HybridRetrievalService


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


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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
        hybrid_retrieval: HybridRetrievalService | None = None,
        context_builder: ContextBuilder | None = None,
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
        self.hybrid_retrieval = hybrid_retrieval
        self.context_builder = context_builder
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
        root = Path(project.repo_root).resolve()
        for pf in plan_files.files:
            target = resolve_within(root, pf.path)
            current = target.read_bytes().decode("utf-8") if target.exists() else ""
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
        context_bundle: ContextBundle | None = None,
    ) -> tuple[PatchPlan, int]:
        feedback = ""
        for attempt in range(1, _MAX_GENERATION_ATTEMPTS + 1):
            log(f"Generation attempt {attempt}/{_MAX_GENERATION_ATTEMPTS}")
            finops = None
            try:
                files, finops = self._finops_prepare(
                    project, story, relevant_files, static_analysis, feedback, context_bundle
                )
                proposal = self.generation_provider.generate_patch_plan(
                    project,
                    story,
                    files,
                    static_analysis,
                    log,
                    validation_feedback=feedback,
                )
            except FinOpsStoppedError as exc:
                self._record_llm_invocation(
                    run_id, log, status="stopped", error=str(exc), finops=exc.finops
                )
                raise
            except Exception:
                self._record_llm_invocation(
                    run_id, log, status="failed", error=_LLM_INVOCATION_ERROR, finops=finops
                )
                raise
            usage = getattr(self.generation_provider, "last_usage", None)
            self._record_llm_invocation(
                run_id, log, status="ok", usage=usage, finops=finops
            )
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

    def _finops_prepare(
        self,
        project,
        story: UserStory,
        relevant_files: list[tuple[str, str]],
        static_analysis: str,
        validation_feedback: str,
        context_bundle: ContextBundle | None,
    ) -> tuple[list[tuple[str, str]], dict]:
        """Build the ACTUAL final prompt, count it, enforce the effective
        input budget and reduce context boundedly before any provider call.

        Returns (files_to_send, finops_trace). Raises FinOpsStoppedError
        when the context cannot be reduced under budget (no provider call).
        """
        cfg = self.config
        provider = self.generation_provider
        limits = [
            cfg.finops_ticket_token_limit,
            cfg.finops_project_token_limit,
            cfg.finops_model_token_limit,
        ]
        budget = compute_effective_budget(limits)
        max_rounds = max(0, cfg.finops_max_reduction_rounds)
        model = getattr(provider, "model_identity", "") or ""
        max_output = getattr(provider, "max_output_tokens", None)

        files = list(relevant_files)
        rounds = 0
        reduced_paths: list[str] = []

        def _stop_trace(method: str, count: int, result: str) -> dict:
            return {
                "provider": provider.provider_name,
                "model": model,
                "count_method": method,
                "estimated_input_tokens": count,
                "effective_input_budget": budget,
                "threshold_result": result,
                "reduction_outcome": "; ".join(
                    f"dropped {path}" for path in reduced_paths
                )
                or "none",
                "max_output_tokens": max_output,
                "routing_reason": "default configured provider/model",
            }

        while True:
            prompt = provider.build_prompt(
                project, story, files, static_analysis, validation_feedback
            )
            count, method = count_prompt(provider, prompt)
            result = evaluate(count, budget)
            if result == "ok":
                break
            if result == "not_configured":
                break
            if rounds >= max_rounds or context_bundle is None or len(files) <= 1:
                raise FinOpsStoppedError(
                    f"FINOPS_STOP: estimated input tokens {count} exceed effective "
                    f"budget {budget}; context cannot be reduced further "
                    f"(reduction rounds exhausted)",
                    finops=_stop_trace(method, count, result),
                )
            drop_order = file_drop_order(context_bundle.items)
            to_drop = next(
                (path for path in drop_order if any(p == path for p, _ in files)),
                None,
            )
            if to_drop is None:
                raise FinOpsStoppedError(
                    f"FINOPS_STOP: estimated input tokens {count} exceed effective "
                    f"budget {budget}; no further low-value context to drop",
                    finops=_stop_trace(method, count, result),
                )
            files = [(p, c) for p, c in files if p != to_drop]
            reduced_paths.append(to_drop)
            rounds += 1

        if reduced_paths:
            reduction_outcome = "; ".join(f"dropped {path}" for path in reduced_paths)
        elif result == "not_configured":
            reduction_outcome = "no_limits"
        else:
            reduction_outcome = "none"

        costs = compute_costs(
            cfg.finops_pricing, provider.provider_name, model, count, None, max_output
        )
        routing_reason = "default configured provider/model"
        selected_context = self._sanitized_context(context_bundle)
        trace = {
            "provider": provider.provider_name,
            "model": model,
            "count_method": method,
            "estimated_input_tokens": count,
            "effective_input_budget": budget,
            "threshold_result": result,
            "reduction_outcome": reduction_outcome,
            "max_output_tokens": max_output,
            "estimated_max_cost": costs["estimated_max_cost"],
            "actual_operational_cost_estimate": costs["actual_operational_cost_estimate"],
            "routing_reason": routing_reason,
            "selected_context": selected_context,
            "prompt_hash": _sha256(prompt),
            "reduction_rounds": rounds,
        }
        return files, trace

    @staticmethod
    def _sanitized_context(bundle: ContextBundle | None) -> dict | None:
        """Persistable context summary: paths + evidence, never source code."""
        if bundle is None:
            return None
        return {
            "selected_files": list(bundle.selected_files),
            "items": [
                {
                    "qualified_name": item.qualified_name,
                    "symbol_type": item.symbol_type,
                    "file_path": item.file_path,
                    "source": item.source,
                    "priority": item.priority,
                    "reasons": list(item.reasons),
                }
                for item in bundle.items
            ],
        }

    def _run(self, project, story: UserStory, log: Callable[[str], None] = _noop) -> PatchResult:
        previous = self._pending_plans.pop(project.id, None)
        if previous is not None:
            _, _, previous_run_id = previous
            self._set_run_status(previous_run_id, "superseded", log)
        run_id = self._start_run(project, story, log)
        try:
            return self._run_lifecycle(project, story, log, run_id)
        except Exception:
            self._fail_run(run_id, log)
            raise

    def _run_lifecycle(self, project, story: UserStory, log: Callable[[str], None], run_id: str | None) -> PatchResult:
        log("Running static pre-analysis…")
        pre_analysis = self.static_analysis.analyze(project)
        if pre_analysis.findings:
            log(f"Pre-analysis: {len(pre_analysis.findings)} hygiene issue(s) detected")

        context_bundle: ContextBundle | None = None
        retrieval_mode = "legacy"

        if self._hybrid_available(project):
            retrieval_mode, context_bundle, relevant_files = self._hybrid_context(
                project, story, log
            )
        else:
            log("Hybrid index unavailable — using legacy semantic context")
            log("Checking semantic index…")
            if not self.semantic_index.exists(project):
                log("Index missing — building from codebase…")
                chunks = self.semantic_index.rebuild(project)
                log(f"Index built: {len(chunks)} chunks embedded")

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
            version = self._latest_version(project)
            context_bundle = build_legacy_bundle(
                relevant_files,
                max_context_files=self.config.max_context_files,
                project_version_id=version["id"] if version else None,
                version_number=version["version_number"] if version else None,
            )

        plan, attempts_used = self._generate_patch_plan(
            project, story, relevant_files, pre_analysis.report, log,
            run_id=run_id, context_bundle=context_bundle,
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
            retrieval_mode=retrieval_mode,
            context=context_bundle,
        )

    # ------------------------------------------------------------------ hybrid context

    def _hybrid_available(self, project) -> bool:
        """Primary path requires the PostgreSQL hybrid index for the project."""
        return (
            self.hybrid_retrieval is not None
            and self.context_builder is not None
            and bool(
                self.hybrid_retrieval.hybrid_index_status(project.id).get(
                    "available"
                )
            )
        )

    def _story_query(self, story: UserStory) -> str:
        """Ticket text embedded/retrieved: title + description + criteria."""
        parts = [story.title, story.description]
        parts.extend(story.acceptanceCriteria or [])
        return "\n".join(part for part in parts if part and part.strip())

    def _latest_version(self, project) -> dict | None:
        """Latest indexed project version, best-effort (None without a store)."""
        if self.store is None:
            return None
        try:
            row = self.store.find_project_by_external_id(project.id)
            return self.store.latest_version(row["id"]) if row is not None else None
        except Exception:
            return None

    def _hybrid_context(self, project, story: UserStory, log: Callable[[str], None]) -> tuple[str, ContextBundle, list[tuple[str, str]]]:
        """Hybrid retrieval → Context Builder → full selected files.

        Never falls back to legacy here: hybrid was already confirmed
        available. Generation-provider errors must propagate (not be masked).
        """
        log("Using hybrid symbol context")
        query_text = self._story_query(story)
        result = self.hybrid_retrieval.retrieve(
            project.id,
            query_text,
            top_k=self.config.hybrid_retrieval_top_k,
            graph_depth=self.config.hybrid_graph_depth,
            max_related=self.config.hybrid_max_related,
        )
        bundle = self.context_builder.build(
            project=project,
            query=query_text,
            retrieval=result,
            estimated_token_budget=self.config.hybrid_context_budget,
            max_context_files=self.config.max_context_files,
        )
        log(
            f"Context: {bundle.included_count} symbol(s) from "
            f"{len(bundle.selected_files)} file(s) "
            f"({bundle.estimated_tokens:,} estimated tokens)"
        )
        relevant_files = [
            (path, bundle.file_contents.get(path, ""))
            for path in bundle.selected_files
        ]
        return "hybrid", bundle, relevant_files

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
        usage: dict | None = None,
        finops: dict | None = None,
    ) -> None:
        if run_id is None or self.store is None:
            return
        try:
            provider = getattr(self.generation_provider, "provider_name", None)
            model = getattr(self.generation_provider, "model_identity", None)
            usage = usage or {}
            self.store.insert_llm_invocation(
                run_id,
                provider=provider,
                model=model,
                status=status,
                error=error,
                prompt_tokens=usage.get("input_tokens"),
                completion_tokens=usage.get("output_tokens"),
                total_tokens=usage.get("total_tokens"),
                count_method=(finops or {}).get("count_method"),
                estimated_input_tokens=(finops or {}).get("estimated_input_tokens"),
                effective_input_budget=(finops or {}).get("effective_input_budget"),
                threshold_result=(finops or {}).get("threshold_result"),
                reduction_outcome=(finops or {}).get("reduction_outcome"),
                max_output_tokens=(finops or {}).get("max_output_tokens"),
                estimated_max_cost=(finops or {}).get("estimated_max_cost"),
                actual_operational_cost_estimate=(finops or {}).get(
                    "actual_operational_cost_estimate"
                ),
                routing_reason=(finops or {}).get("routing_reason"),
                selected_context=(finops or {}).get("selected_context"),
                prompt_hash=(finops or {}).get("prompt_hash"),
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
