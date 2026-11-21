"""Text-to-agent-to-text stage; audio stages will wrap this."""
from collections.abc import Callable

from ..agents.codex import CodexAgent
from .learned_commands import LearnedCommands
from .session import Session


class Pipeline:
    def __init__(self, agent, max_history_turns: int = 8, *, max_history_chars: int = 4000):
        self.agent = agent
        self.session = Session(max_history_turns, max_history_chars=max_history_chars)

    def respond(
        self, text: str, *, spoken: bool = False,
        on_progress: Callable[[str], None] | None = None,
        learned_commands: LearnedCommands | None = None,
    ) -> str:
        text = text.strip()
        if not text:
            raise ValueError("Message cannot be empty")
        prompt = self.session.prompt(text, spoken=spoken)
        candidates = learned_commands.candidates(text) if learned_commands is not None else {}
        if candidates:
            prompt = f"{learned_commands.instructions(candidates)}\n\n{prompt}"
        if on_progress is not None and isinstance(self.agent, CodexAgent):
            response = self.agent.ask(prompt, on_progress=on_progress)
        else:
            response = self.agent.ask(prompt)
        if candidates:
            response = learned_commands.finish(
                text, response, candidates, self.agent.cwd, self.agent.timeout_seconds,
            )
        self.session.remember(text, response)
        return response
