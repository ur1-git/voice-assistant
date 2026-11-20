"""Shared CLI execution helpers."""
from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess


class AgentError(RuntimeError):
    """CLI unavailable, failed, or returned invalid output."""


@dataclass
class Agent:
    executable: str
    cwd: Path
    timeout_seconds: int = 180

    def ensure_available(self) -> None:
        if not shutil.which(self.executable):
            raise AgentError(f"Executable '{self.executable}' not found in PATH")

    def run_process(self, args: list[str], *, stdin: str | None = None) -> str:
        self.ensure_available()
        try:
            result = subprocess.run(
                args,
                input=stdin,
                text=True,
                capture_output=True,
                cwd=self.cwd,
                timeout=self.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise AgentError(f"Agent exceeded {self.timeout_seconds}s timeout") from exc
        except OSError as exc:
            raise AgentError(f"Couldn't start {self.executable}: {exc}") from exc
        if result.returncode != 0:
            message = (result.stderr or result.stdout).strip()[-1200:]
            raise AgentError(f"{self.executable} exited {result.returncode}: {message}")
        return result.stdout
