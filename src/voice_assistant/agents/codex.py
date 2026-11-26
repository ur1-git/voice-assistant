"""Run Codex and extract its final reply from JSON events."""
from dataclasses import dataclass
import json
from collections.abc import Callable
from time import monotonic
from .base import Agent, AgentError
from .progress import stream_codex


@dataclass
class CodexAgent(Agent):
    full_access: bool = False
    model: str | None = None
    reasoning_effort: str | None = None

    def ask(
        self, prompt: str, *, on_progress: Callable[[str], None] | None = None,
        check_interruption: Callable[[], None] | None = None,
    ) -> str:
        # '-' feeds stdin; no shell invocation or command interpolation.
        if self.full_access:
            permissions = ["--dangerously-bypass-approvals-and-sandbox"]
            access = (
                "You are running locally as the user's Linux account with Codex's sandbox disabled. "
                "Your working directory is your starting location. You can use tools to read and "
                "write elsewhere, including Downloads, subject to the account's permissions. "
                "For Downloads, use `xdg-user-dir DOWNLOAD` if available, or `$HOME/Downloads`."
            )
        else:
            permissions = ["--sandbox", "read-only"]
            access = "You are running locally in Codex's read-only sandbox. You can use tools to inspect files."
        prompt = (
            f"Local tool access: {access}\n"
            "When the user asks you to perform a local action, do it in this turn if permitted. "
            "Check access with tools when unsure. If blocked, explain the specific permission or tool error. "
            "Verify the result before reporting success.\n\n"
            f"{prompt}"
        )
        args = [
            self.executable, "exec", "--json", *permissions, "--skip-git-repo-check",
        ]
        if self.model is not None:
            args.extend(["--model", self.model])
        if self.reasoning_effort is not None:
            args.extend(["--config", f"model_reasoning_effort={json.dumps(self.reasoning_effort)}"])
        args.append("-")
        if on_progress is None:
            output = self.run_process(args, stdin=prompt, check_interruption=check_interruption)
        else:
            output = stream_codex(
                self, args, prompt, progress_events(on_progress), check_interruption=check_interruption,
            )
        messages = []
        errors = []
        for line in output.splitlines():
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "item.completed":
                item = event.get("item", {})
                if item.get("type") == "agent_message" and isinstance(item.get("text"), str):
                    messages.append(item["text"])
            elif event.get("type") == "turn.failed":
                errors.append(str(event.get("error", "unknown error")))
        if errors:
            raise AgentError("Codex turn failed: " + "; ".join(errors))
        if not messages:
            raise AgentError("Codex returned no final agent message")
        return messages[-1].strip()


def progress_events(announce: Callable[[str], None]) -> Callable[[str | None], None]:
    last_update = monotonic()
    heartbeat_sent = False
    last_message = ""

    def on_event(line: str | None) -> None:
        nonlocal last_update, heartbeat_sent, last_message
        elapsed = monotonic() - last_update
        if line is None:
            if heartbeat_sent or elapsed < 15:
                return
            heartbeat_sent = True
            message = "I’m still working on it."
        else:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                return
            if event.get("type") == "turn.started":
                message = "I heard you."
            elif event.get("type") == "item.started" and elapsed >= 8:
                message = {
                    "command_execution": "I’m running a command.",
                    "file_change": "I’m updating files.",
                    "web_search": "I’m looking that up.",
                }.get(event.get("item", {}).get("type"))
            else:
                return
        if message and message != last_message:
            announce(message)
            last_message = message
            last_update = monotonic()

    return on_event
