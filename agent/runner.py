from __future__ import annotations

from agent.config import ProjectConfig
from agent.models import TestRunResult
from agent.safe_runner import SafeCommandExecutor, split_command


class TestRunner:
    def run_tests(self, project: ProjectConfig) -> TestRunResult:
        return self._run(project.repo_root, project.test_command, project.validation_timeout_seconds)

    def run_static_analysis(self, project: ProjectConfig) -> TestRunResult:
        cmd = project.static_analysis_command
        if not cmd or not cmd.strip():
            return TestRunResult(command="", exit_code=0, output="Static analysis disabled")
        return self._run(project.repo_root, cmd, project.validation_timeout_seconds)

    def _run(self, working_dir: str, command: str, timeout: int = 300) -> TestRunResult:
        if not command or not command.strip():
            return TestRunResult(command=command, exit_code=0, output="No command specified")
        try:
            argv = split_command(command)
            return SafeCommandExecutor().run(argv, cwd=working_dir, timeout=timeout)
        except Exception as exc:
            return TestRunResult(command=command, exit_code=1, output=f"Execution error: {exc}")
