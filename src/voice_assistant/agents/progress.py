"""Read Codex events while its subprocess is running."""
from collections.abc import Callable
import os
from selectors import DefaultSelector, EVENT_READ
import subprocess
from tempfile import TemporaryFile
from time import monotonic

from .base import Agent, AgentError, agent_process


def stream_codex(
    agent: Agent, args: list[str], prompt: str, on_event: Callable[[str | None], None],
    *, check_interruption: Callable[[], None] | None = None,
) -> str:
    agent.ensure_available()
    try:
        with TemporaryFile() as request, TemporaryFile() as errors:
            request.write(prompt.encode())
            request.seek(0)
            if check_interruption is not None:
                check_interruption()
            with agent_process(
                args, interruptible=check_interruption is not None,
                stdin=request, stdout=subprocess.PIPE, stderr=errors, cwd=agent.cwd,
            ) as process:
                deadline = monotonic() + agent.timeout_seconds
                output = bytearray()
                pending = b""
                with DefaultSelector() as selector:
                    selector.register(process.stdout, EVENT_READ)
                    while True:
                        if check_interruption is not None:
                            check_interruption()
                        remaining = deadline - monotonic()
                        if remaining <= 0:
                            raise subprocess.TimeoutExpired(args, agent.timeout_seconds)
                        if not selector.select(min(remaining, 0.1 if check_interruption else 1)):
                            on_event(None)
                            continue
                        chunk = os.read(process.stdout.fileno(), 65536)
                        if not chunk:
                            if process.poll() is not None:
                                break
                            try:
                                process.wait(timeout=min(remaining, 0.1))
                            except subprocess.TimeoutExpired:
                                continue
                            break
                        output.extend(chunk)
                        *lines, pending = (pending + chunk).split(b"\n")
                        for line in lines:
                            if check_interruption is not None:
                                check_interruption()
                            on_event(line.decode(errors="replace"))
                        on_event(None)
                if check_interruption is not None:
                    check_interruption()
            if process.returncode:
                errors.seek(0)
                detail = (errors.read() or output).decode(errors="replace").strip()[-1200:]
                raise AgentError(f"{agent.executable} exited {process.returncode}: {detail}")
            return output.decode(errors="replace")
    except subprocess.TimeoutExpired as exc:
        raise AgentError(f"Agent exceeded {agent.timeout_seconds}s timeout") from exc
    except OSError as exc:
        raise AgentError(f"Couldn't run {agent.executable}: {exc}") from exc
