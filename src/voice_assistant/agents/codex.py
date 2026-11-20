"""Codex CLI JSON event streaming, executed with read-only sandbox."""
import json
from .base import Agent, AgentError


class CodexAgent(Agent):
    def ask(self, prompt: str) -> str:
        # '-' feeds stdin; no shell invocation or command interpolation.
        output = self.run_process([
            self.executable, "exec", "--json", "--sandbox", "read-only",
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
