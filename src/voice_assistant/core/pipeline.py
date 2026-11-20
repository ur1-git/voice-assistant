"""Text-to-agent-to-text stage; audio stages will wrap this."""
from .session import Session


class Pipeline:
    def __init__(self, agent, max_history_turns: int = 8, *, max_history_chars: int = 4000):
        self.agent = agent
        self.session = Session(max_history_turns, max_history_chars=max_history_chars)

    def respond(self, text: str, *, spoken: bool = False) -> str:
        text = text.strip()
        if not text:
            raise ValueError("Message cannot be empty")
        response = self.agent.ask(self.session.prompt(text, spoken=spoken))
        self.session.remember(text, response)
        return response
