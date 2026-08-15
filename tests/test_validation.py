"""F5 validation contract tests (#37/#39/#41/#43/#44) — state machine + selection."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

from agent.config import ProjectConfig
from agent.models import PatchFile, PatchPlan
from agent.validation import (
    ValidationWorkspace,
    build_maven_argv,
    select_targeted_tests,
)


def _project(root: str) -> ProjectConfig:
    return ProjectConfig(
        id="val-test",
        name="val-test",
        repo_root=root,
        stories_file="",
        index_file="",
    )


def _plan_with(files: list[PatchFile]) -> PatchPlan:
    return PatchPlan(storyId="s1", summary="test", files=files, tests=[], notes=[])


def test_workspace_applies_proposal_without_touching_reference(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    target = source / "pkg" / "Foo.java"
    target.parent.mkdir(parents=True)
    target.write_text("public class Foo { String x = \"NOT_FOUND\"; }\n")
    project = _project(str(source))

    plan = _plan_with([
        PatchFile(
            path="pkg/Foo.java",
            operation="modify",
            content='public class Foo { String x = "RESOURCE_NOT_FOUND"; }\n',
        ),
    ])

    ws = ValidationWorkspace(project)
    try:
        ws.prepare(project)
        assert (ws.root / "pkg" / "Foo.java").exists()
        ws.apply(plan)
        assert "RESOURCE_NOT_FOUND" in (ws.root / "pkg" / "Foo.java").read_text()
        assert "NOT_FOUND" in target.read_text()
    finally:
        ws.cleanup()
    assert not ws.root.exists()
    assert target.exists()


def test_workspace_cleanup_on_exception(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    (source / "Foo.java").write_text("class Foo {}\n")
    project = _project(str(source))
    ws = ValidationWorkspace(project)
    ws.prepare(project)
    assert ws.root.exists()
    ws.cleanup()
    assert not ws.root.exists()


def test_targeted_tests_naming_convention(tmp_path):
    project = _project(str(tmp_path))
    base = tmp_path / "shoppoc-catalog" / "src" / "main" / "java" / "com" / "shoppoc" / "catalog" / "domain"
    base.mkdir(parents=True)
    (base / "Product.java").write_text("x")
    test_file = tmp_path / "shoppoc-catalog" / "src" / "test" / "java" / "com" / "shoppoc" / "catalog" / "domain" / "ProductTest.java"
    test_file.parent.mkdir(parents=True)
    test_file.write_text("x")
    plan = _plan_with([
        PatchFile(path="shoppoc-catalog/src/main/java/com/shoppoc/catalog/domain/Product.java",
                  operation="modify", content="x"),
        PatchFile(path="pkg/README.md", operation="create", content="x"),
        PatchFile(path="pkg/DomainErrorTest.java", operation="create", content="x"),
    ])
    tests = select_targeted_tests(project, plan, store=None, project_version_id=None)
    assert "ProductTest" in tests
    assert "ProductTests" not in tests  # candidate must exist on disk
    assert not any("DomainErrorTest" in t for t in tests)  # created files skipped
    assert all(not t.endswith(".java") for t in tests)


def test_maven_argv_targeted_vs_none():
    project = _project("/tmp/nonexistent")
    argv = build_maven_argv(project, ["MoneyTest", "ProductTest"])
    assert argv == ["mvn", "-q", "-Dtest=MoneyTest,ProductTest", "test"]
    assert build_maven_argv(project, []) is None


class _NoopStore:
    def tests_for_paths(self, version_id, paths):
        return ["GlobalExceptionHandlerTest"]


def test_targeted_tests_graph_edges_preferred(tmp_path):
    project = _project(str(tmp_path))
    (tmp_path / "shoppoc-shared" / "src" / "main" / "java" / "com" / "shoppoc" / "shared" / "error").mkdir(parents=True)
    (tmp_path / "shoppoc-shared" / "src" / "main" / "java" / "com" / "shoppoc" / "shared" / "error" / "DomainError.java").write_text("x")
    plan = _plan_with([
        PatchFile(path="shoppoc-shared/src/main/java/com/shoppoc/shared/error/DomainError.java",
                  operation="modify", content="x"),
    ])
    tests = select_targeted_tests(project, plan, store=_NoopStore(), project_version_id="v1")
    assert "GlobalExceptionHandlerTest" in tests


def _accept_gate_orchestrator():
    from agent.orchestrator import AgentOrchestrator

    orch = AgentOrchestrator.__new__(AgentOrchestrator)
    orch.config = None
    orch.patch_applier = None
    orch.store = None
    orch.generation_provider = None
    orch._pending_plans = {}
    return orch


def test_accept_requires_validated_pass():
    orch = _accept_gate_orchestrator()
    plan = _plan_with([PatchFile(path="a.java", operation="modify", content="x")])
    orch._pending_plans["p1"] = (plan, "diff", None, "failed")
    with pytest.raises(ValueError, match="VALIDATION_REQUIRED"):
        orch.accept_plan("p1")
    # plan stays parked so the human can still reject
    assert "p1" in orch._pending_plans


def test_accept_allowed_when_validation_passed_or_skipped():
    from agent.patch import PatchApplierService

    class _FakeApplier:
        def apply(self, project, plan):
            pass

    orch = _accept_gate_orchestrator()
    orch.patch_applier = _FakeApplier()
    orch.config = type("C", (), {"project_by_id": staticmethod(lambda pid: None)})()
    plan = _plan_with([PatchFile(path="a.java", operation="modify", content="x")])
    orch._pending_plans["p1"] = (plan, "diff", None, "passed")
    assert orch.accept_plan("p1") == "diff"
    assert "p1" not in orch._pending_plans

    orch._pending_plans["p2"] = (plan, "diff", None, None)
    assert orch.accept_plan("p2") == "diff"
