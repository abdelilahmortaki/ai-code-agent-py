from __future__ import annotations
import os
import shlex
import subprocess
from pathlib import Path

from agent.models import TestRunResult

_CMD_EXE_METACHARS = set("&|<>^%")


def _windows_split(command: str) -> list[str]:
    """Deterministic Windows command-line tokenizer (CommandLineToArgvW rules).

    Double quotes group whitespace; backslash escapes a following quote;
    runs of backslashes before a quote are halved; ``""`` inside quotes is a
    literal quote. Quotes are consumed by parsing and never passed through
    to the child process. No shell is involved.
    """
    args: list[str] = []
    arg: list[str] = []
    started = False
    in_quotes = False
    i, n = 0, len(command)
    while i < n:
        ch = command[i]
        if ch == "\\":
            j = i
            while j < n and command[j] == "\\":
                j += 1
            count = j - i
            if j < n and command[j] == '"':
                arg.append("\\" * (count // 2))
                if count % 2 == 1:
                    arg.append('"')
                    started = True
                    i = j + 1
                    continue
                in_quotes = not in_quotes
                started = True
                i = j + 1
                continue
            arg.append("\\" * count)
            started = True
            i = j
        elif ch == '"':
            if in_quotes and i + 1 < n and command[i + 1] == '"':
                arg.append('"')
                started = True
                i += 2
                continue
            in_quotes = not in_quotes
            started = True
            i += 1
        elif ch in " \t" and not in_quotes:
            if started:
                args.append("".join(arg))
                arg = []
                started = False
            i += 1
        else:
            arg.append(ch)
            started = True
            i += 1
    if started:
        args.append("".join(arg))
    return args


def split_command(command: str) -> list[str]:
    if os.name == "nt":
        for ch in command:
            if ch in _CMD_EXE_METACHARS:
                raise ValueError(f"Shell metacharacter not allowed in command: {ch!r}")
        return _windows_split(command)
    return shlex.split(command)


class SafeCommandExecutor:
    def run(self, argv: list[str], cwd: str | Path, timeout: int) -> TestRunResult:
        if not argv:
            raise ValueError("Empty command")
        if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout <= 0:
            raise ValueError(f"Timeout must be a positive integer, got {timeout!r}")
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
