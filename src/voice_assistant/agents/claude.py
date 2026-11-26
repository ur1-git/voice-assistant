"""Claude Code one-shot, with tools disabled for the prototype."""
from collections.abc import Callable
import json
from .base import Agent, AgentError


class ClaudeAgent(Agent):
    def ask(self, prompt: str, *, check_interruption: Callable[[], None] | None = None) -> str:
        # --tools "" disables Claude's tool access in this early prototype.
        output = self.run_process([
            self.executable, "-p", "--output-format", "json",
            "--tools", "", "--", prompt,
        ], check_interruption=check_interruption)
        try:
            data = json.loads(output)
            response = data.get("result")
            if data.get("is_error") or not isinstance(response, str) or not response.strip():
                raise ValueError("Missing response or agent returned an error")
            return response.strip()
        except (json.JSONDecodeError, ValueError) as exc:
            raise AgentError(f"Invalid Claude response: {exc}") from exc
