"""Small in-memory session; persistent provider sessions come later."""
from dataclasses import dataclass, field


@dataclass
class Session:
    max_turns: int = 8
    history: list[tuple[str, str]] = field(default_factory=list)

    def prompt(self, text: str) -> str:
        if not self.history:
            return text
        lines = ["Previous conversation (context only):"]
        for question, answer in self.history[-self.max_turns:] if self.max_turns else []:
            lines.extend((f"User: {question}", f"Assistant: {answer}"))
        lines.extend(("", "Current user message:", text))
        return "\n".join(lines)

    def remember(self, question: str, answer: str) -> None:
        if not self.max_turns:
            return
        self.history.append((question, answer))
        del self.history[:-self.max_turns]
