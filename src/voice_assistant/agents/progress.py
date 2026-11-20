"""Read Codex events while its subprocess is running."""
from collections.abc import Callable
import os
from selectors import DefaultSelector, EVENT_READ
import subprocess
from tempfile import TemporaryFile
from time import monotonic

from .base import Agent, AgentError


def stream_codex(agent: Agent, args: list[str], prompt: str, on_event: Callable[[str | None], None]) -> str:
    agent.ensure_available()
    try:
        with TemporaryFile() as request, TemporaryFile() as errors:
            request.write(prompt.encode())
            request.seek(0)
            with subprocess.Popen(args, stdin=request, stdout=subprocess.PIPE, stderr=errors, cwd=agent.cwd) as process:
                deadline = monotonic() + agent.timeout_seconds
                output = bytearray()
                pending = b""
                try:
                    with DefaultSelector() as selector:
                        selector.register(process.stdout, EVENT_READ)
                        while True:
                            remaining = deadline - monotonic()
                            if remaining <= 0:
                                raise subprocess.TimeoutExpired(args, agent.timeout_seconds)
                            if not selector.select(min(remaining, 1)):
                                on_event(None)
                                continue
                            chunk = os.read(process.stdout.fileno(), 65536)
                            if not chunk:
                                break
                            output.extend(chunk)
                            *lines, pending = (pending + chunk).split(b"\n")
                            for line in lines:
                                on_event(line.decode(errors="replace"))
                            on_event(None)
                    process.wait(timeout=max(0, deadline - monotonic()))
                finally:
                    if process.poll() is None:
                        process.kill()
            if process.returncode:
                errors.seek(0)
                detail = (errors.read() or output).decode(errors="replace").strip()[-1200:]
                raise AgentError(f"{agent.executable} exited {process.returncode}: {detail}")
            return output.decode(errors="replace")
    except subprocess.TimeoutExpired as exc:
        raise AgentError(f"Agent exceeded {agent.timeout_seconds}s timeout") from exc
    except OSError as exc:
        raise AgentError(f"Couldn't run {agent.executable}: {exc}") from exc
