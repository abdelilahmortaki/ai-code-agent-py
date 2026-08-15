"""F5 disposable validation workspace + targeted tests + bounded repair.

NO Docker. A disposable copy of the reference project is created under a
temp directory, the approved proposal is applied there, targeted tests run
there via the shell-free SafeCommandExecutor, and the workspace is cleaned
up on every path (PASS, FAIL, repair failure, exception, timeout). The
reference project source is never modified.

Targeted test selection:
1. TESTED_BY graph edges of the modified symbols/files (when a store exists);
2. naming convention: <ClassName>Test / <ClassName>Tests for modified classes;
3. fallback: none (validation reported as passed with a recorded reason) or
   the project's full test command, per configuration.
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agent.config import ProjectConfig
from agent.models import PatchPlan, TestRunResult
from agent.paths import resolve_within
from agent.runner import TestRunner

if TYPE_CHECKING:
    from agent.db.store import PgStore


_TEST_NAME = re.compile(r"^([A-Z][A-Za-z0-9]*)\.java$")


def _test_exists(project: ProjectConfig, class_name: str) -> bool:
    """True when a Java file named <class_name>.java exists under the project."""
    root = Path(project.repo_root).resolve()
    if not root.is_dir():
        return False
    for path in root.rglob(f"{class_name}.java"):
        rel_parts = set(path.relative_to(root).parts)
        if rel_parts & {".git", "target", "node_modules"}:
            continue
        return True
    return False


def select_targeted_tests(
    project: ProjectConfig,
    plan: PatchPlan,
    store: "PgStore | None" = None,
    project_version_id: str | None = None,
) -> list[str]:
    """Deterministic targeted test selection for the modified files.

    Only files that already exist in the reference project are considered
    (create operations are skipped — a proposed brand-new file cannot have
    an existing test), and naming-convention candidates must actually exist
    so Maven never fails on a guessed test name.
    """
    modified = [
        f.path
        for f in plan.files
        if f.operation == "modify"
        and (Path(project.repo_root) / f.path).is_file()
    ]
    tests: list[str] = []
    seen: set[str] = set()

    def add(name: str) -> None:
        if name and name not in seen:
            seen.add(name)
            tests.append(name)

    # 1. TESTED_BY graph edges (store-backed).
    if store is not None and project_version_id and modified:
        try:
            for path in modified:
                for test_name in store.tests_for_paths(
                    project_version_id, [path]
                ):
                    add(test_name)
        except Exception:
            pass

    # 2. Naming convention for modified Java files (must exist on disk).
    for path in modified:
        if not path.endswith(".java"):
            continue
        base = Path(path).name
        match = _TEST_NAME.match(base)
        if not match:
            continue
        for candidate in (match.group(1) + "Test", match.group(1) + "Tests"):
            if _test_exists(project, candidate):
                add(candidate)

    return tests


def build_maven_argv(project: ProjectConfig, tests: list[str]) -> list[str]:
    """Return a real shell-free Maven command for every validation path.

    Targeted tests run via ``-Dtest=`` on the reactor (all modules compile in
    dependency order, only the selected tests execute). When no targeted
    tests exist, ``verify`` runs; a configured command containing ``verify``
    is honored as the repository-defined variant.
    """
    executable = "mvn.cmd" if os.name == "nt" else "mvn"
    if tests:
        return [
            executable,
            "-q",
            f"-Dtest={','.join(tests)}",
            "-DfailIfNoTests=false",
            "test",
        ]
    from agent.safe_runner import split_command

    configured = (project.test_command or "").strip()
    if configured:
        try:
            configured_argv = split_command(configured)
        except ValueError:
            configured_argv = []
        if "verify" in configured_argv:
            if os.name == "nt" and configured_argv[0].casefold() == "mvn":
                configured_argv[0] = executable
            return configured_argv
    return [executable, "-q", "verify"]


class ValidationWorkspace:
    """Disposable copy of a project; proposal applied there; always cleaned up."""

    def __init__(self, project: ProjectConfig) -> None:
        self._tmp = tempfile.mkdtemp(prefix="agent-validation-")
        self.root = Path(self._tmp)
        self._copied = False

    def prepare(self, project: ProjectConfig) -> Path:
        shutil.copytree(
            project.repo_root,
            self.root,
            ignore=shutil.ignore_patterns(".git", "target", ".agent", "node_modules"),
            dirs_exist_ok=True,
        )
        self._copied = True
        return self.root

    def apply(self, plan: PatchPlan) -> None:
        """Write the proposal into the workspace copy (not the reference)."""
        for file_entry in plan.files:
            target = resolve_within(self.root, file_entry.path)
            if file_entry.operation == "delete":
                if target.exists():
                    target.unlink()
                continue
            if file_entry.operation == "create" and target.exists():
                raise ValueError("Create target already exists in workspace")
            target.parent.mkdir(parents=True, exist_ok=True)
            content = file_entry.content or ""
            target.write_bytes(content.encode("utf-8"))

    def cleanup(self) -> None:
        shutil.rmtree(self._tmp, ignore_errors=True)


class ValidationService:
    """Runs validation in a disposable workspace and manages bounded repair."""

    def __init__(
        self,
        test_runner: TestRunner,
        store: "PgStore | None" = None,
        max_repair_attempts: int = 2,
        fallback: str = "none",
    ) -> None:
        self.test_runner = test_runner
        self.store = store
        self.max_repair_attempts = max(0, int(max_repair_attempts))
        self.fallback = fallback

    def validate(self, project: ProjectConfig, plan: PatchPlan) -> TestRunResult:
        """Run targeted tests against a disposable copy of the project.

        Returns the TestRunResult; ``command`` is empty when no tests were
        selected and the fallback is disabled (validation effectively PASS).
        """
        workspace = ValidationWorkspace(project)
        try:
            workspace.prepare(project)
            workspace.apply(plan)
            project_version_id = self._latest_version_id(project)
            tests = select_targeted_tests(
                project, plan, self.store, project_version_id
            )
            argv = build_maven_argv(project, tests)
            command = " ".join(argv)
            result = self.test_runner._run(
                str(workspace.root), command, project.validation_timeout_seconds
            )
            return result.model_copy(update={
                "strategy": "targeted" if tests else "fallback",
                "targeted_tests": tests,
            })
        finally:
            workspace.cleanup()

    def _latest_version_id(self, project: ProjectConfig) -> str | None:
        if self.store is None:
            return None
        try:
            row = self.store.find_project_by_external_id(project.id)
            version = self.store.latest_version(row["id"]) if row is not None else None
            return version["id"] if version else None
        except Exception:
            return None


def build_failure_context(
    story_id: str,
    summary: str,
    git_diff: str,
    test_result: TestRunResult,
    static_analysis_report: str,
    attempt_number: int,
) -> Any:
    """Build a TestFailureContext for the repair generation."""
    from agent.models import TestFailureContext

    return TestFailureContext(
        story_id=story_id,
        summary=summary,
        git_diff=git_diff,
        test_command=test_result.command,
        test_output=test_result.output[-12000:],
        static_analysis_report=static_analysis_report,
        attempt_number=attempt_number,
    )
