from __future__ import annotations
import subprocess
from pathlib import Path

from agent.config import ProjectConfig
from agent.models import TestRunResult


class TestRunner:
    def run_tests(self, project: ProjectConfig) -> TestRunResult:
        return self._run(project.repo_root, project.test_command)

    def run_static_analysis(self, project: ProjectConfig) -> TestRunResult:
        cmd = project.static_analysis_command
        if not cmd or not cmd.strip():
            return TestRunResult(command="", exit_code=0, output="Static analysis disabled")
        return self._run(project.repo_root, cmd)

    def _run(self, working_dir: str, command: str) -> TestRunResult:
        if not command or not command.strip():
            return TestRunResult(command=command, exit_code=0, output="No command specified")
        try:
            result = subprocess.run(
                command,
                cwd=Path(working_dir).resolve(),
                shell=True,          # handles mvn, sh, cmd.exe transparently
                capture_output=True,
                text=True,
                timeout=300,
            )
            output = (result.stdout or "") + (result.stderr or "")
            return TestRunResult(command=command, exit_code=result.returncode, output=output)
        except subprocess.TimeoutExpired:
            return TestRunResult(command=command, exit_code=1, output="Command timed out after 300s")
        except Exception as e:
            return TestRunResult(command=command, exit_code=1, output=f"Execution error: {e}")
