"""Focused FinOps contract tests (F3.3-F3.7) — pure functions only."""

from __future__ import annotations

from agent.finops import (
    compute_costs,
    compute_effective_budget,
    count_prompt,
    estimate_tokens,
    evaluate,
    file_drop_order,
)


def test_estimate_tokens_chars_per_4():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcdefgh") == 2


class _CountingProvider:
    provider_name = "fake"

    def count_tokens(self, text):  # noqa: D401
        return 7


class _NonCountingProvider:
    provider_name = "fake"

    def count_tokens(self, text):  # noqa: D401
        return None


def test_count_prompt_exact_vs_estimate():
    exact, method = count_prompt(_CountingProvider(), "hello world")
    assert exact == 7 and method == "exact:provider"
    est, method = count_prompt(_NonCountingProvider(), "abcdefgh")
    assert est == 2 and method == "estimate:chars/4"


def test_count_prompt_passes_effective_model_identity():
    class Provider:
        def count_tokens(self, text, model_or_deployment=None):
            assert model_or_deployment == "routed-model"
            return 3

    assert count_prompt(Provider(), "hello", "routed-model") == (3, "exact:provider")


def test_effective_budget_is_min_of_configured_limits():
    assert compute_effective_budget([None, None]) is None
    assert compute_effective_budget([1000, None, 500]) == 500
    assert compute_effective_budget([0, 1000]) == 1000


def test_evaluate_threshold():
    assert evaluate(100, 1000) == "ok"
    assert evaluate(1000, 500) == "over"
    assert evaluate(100, None) == "not_configured"


def test_costs_use_configured_pricing_and_report_unavailable_otherwise():
    pricing = {"azure": {"gpt-5-mini": {"input_per_million": 1.10, "output_per_million": 4.40}}}
    costs = compute_costs(pricing, "azure", "gpt-5-mini", 10_000, 2_000, 8_192)
    assert costs["estimated_max_cost"] is not None
    assert costs["actual_operational_cost_estimate"] is not None
    # 10000 in * 1.10 + 2000 out * 4.40 per million
    assert abs(costs["actual_operational_cost_estimate"] - (0.011 + 0.0088)) < 1e-6
    unavailable = compute_costs(pricing, "azure", "unknown-model", 100, 100, 100)
    assert unavailable["estimated_max_cost"] is None
    assert unavailable["actual_operational_cost_estimate"] is None


class _Item:
    def __init__(self, path, source, score):
        self.file_path = path
        self.source = source
        self.final_score = score
        self.included = True


def test_drop_order_preserves_high_value_evidence_last():
    items = [
        _Item("main.java", "main_candidate", 0.9),
        _Item("dep.java", "dependency", 0.5),
        _Item("ticket.java", "ticket_evidence", 0.7),
    ]
    assert file_drop_order(items) == ["dep.java", "ticket.java", "main.java"]


def test_routed_model_drives_count_and_pricing():
    from agent.config import AgentConfig
    from agent.orchestrator import AgentOrchestrator

    class Provider:
        provider_name = "fake"
        model_identity = "model-a"

        def count_tokens(self, text, model_or_deployment=None):
            assert model_or_deployment == "model-b"
            return 10

    orch = AgentOrchestrator.__new__(AgentOrchestrator)
    orch.config = AgentConfig(finops_pricing={
        "fake": {
            "model-a": {"input_per_million": 1, "output_per_million": 1},
            "model-b": {"input_per_million": 2, "output_per_million": 3},
        }
    })
    orch.generation_provider = Provider()
    trace = orch._finops_trace("prompt", "model-b", 100, "routed", None)
    assert trace["model"] == "model-b"
    assert trace["count_method"] == "exact:provider"
    assert trace["estimated_max_cost"] == 0.00032


def test_repair_finops_stop_precedes_provider_call(tmp_path):
    from agent.config import AgentConfig, ProjectConfig
    from agent.models import PatchFile, PatchPlan, TestRunResult as RunResult, UserStory
    from agent.orchestrator import AgentOrchestrator

    source = tmp_path / "repo"
    source.mkdir()
    (source / "Foo.java").write_text("class Foo {}\n")
    project = ProjectConfig(
        id="repair-test", name="repair-test", repo_root=str(source),
        stories_file="", index_file="",
    )
    plan = PatchPlan(
        storyId="s1", summary="repair", files=[
            PatchFile(path="Foo.java", operation="modify", content="class Foo {}\n")
        ]
    )

    class Provider:
        provider_name = "fake"
        model_identity = "model-a"
        max_output_tokens = 100
        calls = 0

        def build_fix_prompt(self, ctx):
            return "repair prompt"

        def count_tokens(self, text, model_or_deployment=None):
            return 100

        def generate_fix_plan(self, ctx, log, options=None):
            self.calls += 1
            raise AssertionError("blocked repair must not call provider")

    class Runner:
        def _run(self, working_dir, command, timeout):
            return RunResult(command=command, exit_code=1, output="failed", argv=command.split(), executed=True)

    orch = AgentOrchestrator.__new__(AgentOrchestrator)
    orch.config = AgentConfig(
        finops_project_token_limit=1,
        finops_pricing={"fake": {"model-a": {"input_per_million": 1, "output_per_million": 1}}},
        max_repair_attempts=1,
    )
    orch.generation_provider = Provider()
    orch.store = None
    orch.test_runner = Runner()
    orch.materializer = __import__("agent.materializer", fromlist=["PatchMaterializer"]).PatchMaterializer()
    traces = []
    orch._record_llm_invocation = lambda run_id, log, status, error=None, usage=None, finops=None: traces.append((status, finops))
    result = orch._run_validation(
        project, UserStory(id="s1", title="x", description="x"), plan, "diff",
        None, lambda _: None, "", None, None,
    )
    assert result[1] == "failed" and orch.generation_provider.calls == 0
    assert traces[0][0] == "stopped"
    assert traces[0][1]["estimated_input_tokens"] == 100
    assert traces[0][1]["threshold_result"] == "over"
