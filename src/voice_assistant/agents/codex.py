"""Run Codex and extract its final reply from JSON events."""
from dataclasses import dataclass
import json
from .base import Agent, AgentError


@dataclass
class CodexAgent(Agent):
    full_access: bool = False

    def ask(self, prompt: str) -> str:
        # '-' feeds stdin; no shell invocation or command interpolation.
        permissions = (
            ["--dangerously-bypass-approvals-and-sandbox"]
            if self.full_access else ["--sandbox", "read-only"]
        )
        output = self.run_process([
            self.executable, "exec", "--json", *permissions,
            "--skip-git-repo-check", "-",
        ], stdin=prompt)
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
