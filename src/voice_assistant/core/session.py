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
    max_history_chars: int = 4000

    def prompt(self, text: str, *, spoken: bool = False) -> str:
        lines = [SPOKEN_REPLY_RULES, ""] if spoken else []
        history = []
        remaining = self.max_history_chars
        for question, answer in reversed(self.history[-self.max_turns:] if self.max_turns else []):
            if remaining <= 0:
                break
            turn = f"User: {question}\nAssistant: {answer}\n"
            if len(turn) > remaining:
                if not history:
                    history.append(turn[:remaining - 1] + "…")
                break
            history.append(turn)
            remaining -= len(turn)
        if not history:
            return "\n".join([*lines, text])
        lines.append("Previous conversation (context only):")
        lines.append("".join(reversed(history)).rstrip())
        lines.extend(("", "Current user message:", text))
        return "\n".join(lines)

    def remember(self, question: str, answer: str) -> None:
        if not self.max_turns:
            return
        self.history.append((question, answer))
        del self.history[:-self.max_turns]
