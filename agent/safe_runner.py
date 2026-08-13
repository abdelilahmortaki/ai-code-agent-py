from __future__ import annotations
import os
import shlex
import subprocess
from pathlib import Path

from agent.models import TestRunResult

_CMD_EXE_METACHARS = set("&|<>^%")


def split_command(command: str) -> list[str]:
    if os.name == "nt":
        for ch in command:
            if ch in _CMD_EXE_METACHARS:
                raise ValueError(f"Shell metacharacter not allowed in command: {ch!r}")
        return shlex.split(command, posix=False)
    return shlex.split(command)


class SafeCommandExecutor:
    def run(self, argv: list[str], cwd: str | Path, timeout: int) -> TestRunResult:
        if not argv:
            raise ValueError("Empty command")
        workdir = Path(cwd).resolve()
        if not workdir.exists():
            raise ValueError(f"Working directory does not exist: {workdir}")
        if not workdir.is_dir():
            raise ValueError(f"Working directory is not a directory: {workdir}")
        try:
            result = subprocess.run(
                argv,
                cwd=workdir,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            output = (result.stdout or "") + (result.stderr or "")
            return TestRunResult(command=" ".join(argv), exit_code=result.returncode, output=output)
        except subprocess.TimeoutExpired as exc:
            # Py3.10 returns bytes on TimeoutExpired even with text=True
            stdout = exc.stdout or b""
            stderr = exc.stderr or b""
            if isinstance(stdout, bytes):
                stdout = stdout.decode("utf-8", errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors="replace")
            output = stdout + stderr + f"\nCommand timed out after {timeout}s"
            return TestRunResult(command=" ".join(argv), exit_code=124, output=output)
        except Exception as exc:
            return TestRunResult(command=" ".join(argv), exit_code=1, output=f"Execution error: {exc}")
