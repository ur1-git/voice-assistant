"""Text-to-agent-to-text stage; audio stages will wrap this."""
from .session import Session


class Pipeline:
    def __init__(self, agent, max_history_turns: int = 8):
        self.agent = agent
        self.session = Session(max_history_turns)

    def respond(self, text: str) -> str:
        text = text.strip()
        if not text:
            raise ValueError("Message cannot be empty")
        response = self.agent.ask(self.session.prompt(text))
        self.session.remember(text, response)
        return response
