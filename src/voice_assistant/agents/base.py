"""Shared CLI execution helpers."""
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import signal
import subprocess
from time import monotonic


class AgentError(RuntimeError):
    """CLI unavailable, failed, or returned invalid output."""


@contextmanager
def agent_process(args: list[str], *, interruptible: bool, **options):
    with subprocess.Popen(args, start_new_session=interruptible, **options) as process:
        try:
            yield process
        finally:
            if process.poll() is None:
                if interruptible:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                else:
                    process.kill()


@dataclass
class Agent:
    executable: str
    cwd: Path
    timeout_seconds: int = 180

    def ensure_available(self) -> None:
        if not shutil.which(self.executable):
            raise AgentError(f"Executable '{self.executable}' not found in PATH")

    def run_process(
        self, args: list[str], *, stdin: str | None = None,
        check_interruption: Callable[[], None] | None = None,
    ) -> str:
        self.ensure_available()
        try:
            if check_interruption is None:
                result = subprocess.run(
                    args, input=stdin, text=True, capture_output=True, cwd=self.cwd,
                    timeout=self.timeout_seconds, check=False,
                )
            else:
                check_interruption()
                deadline = monotonic() + self.timeout_seconds
                with agent_process(
                    args, interruptible=True, cwd=self.cwd, text=True,
                    stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                ) as process:
                    while True:
                        check_interruption()
                        remaining = deadline - monotonic()
                        if remaining <= 0:
                            raise subprocess.TimeoutExpired(args, self.timeout_seconds)
                        try:
                            stdout, stderr = process.communicate(stdin, timeout=min(0.1, remaining))
                        except subprocess.TimeoutExpired:
                            stdin = None
                            continue
                        check_interruption()
                        result = subprocess.CompletedProcess(args, process.returncode, stdout, stderr)
                        break
        except subprocess.TimeoutExpired as exc:
            raise AgentError(f"Agent exceeded {self.timeout_seconds}s timeout") from exc
        except OSError as exc:
            raise AgentError(f"Couldn't start {self.executable}: {exc}") from exc
        if result.returncode != 0:
            message = (result.stderr or result.stdout).strip()[-1200:]
            raise AgentError(f"{self.executable} exited {result.returncode}: {message}")
        return result.stdout
