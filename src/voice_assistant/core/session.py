"""Small in-memory session; persistent provider sessions come later."""
from dataclasses import dataclass, field


SPOKEN_REPLY_RULES = (
    "Your entire reply will be read aloud. Use short, conversational sentences "
    "in the language of the user's request. Refer to websites by their names "
    "and explain technical terms in everyday language where possible. "
    "Give exact URLs, long paths, commands, code, or lengthy technical names "
    "only if the user explicitly asks for the exact wording. Use plain text "
    "and avoid markdown tables, link syntax, and citation markers."
)


@dataclass
class Session:
    max_turns: int = 8
    history: list[tuple[str, str]] = field(default_factory=list)

    def prompt(self, text: str, *, spoken: bool = False) -> str:
        lines = [SPOKEN_REPLY_RULES, ""] if spoken else []
        if not self.history:
            return "\n".join([*lines, text])
        lines.append("Previous conversation (context only):")
        for question, answer in self.history[-self.max_turns:] if self.max_turns else []:
            lines.extend((f"User: {question}", f"Assistant: {answer}"))
        lines.extend(("", "Current user message:", text))
        return "\n".join(lines)

    def remember(self, question: str, answer: str) -> None:
        if not self.max_turns:
            return
        self.history.append((question, answer))
        del self.history[:-self.max_turns]
