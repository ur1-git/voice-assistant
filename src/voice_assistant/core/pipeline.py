"""Send requests to the agent and track the conversation."""
from collections.abc import Callable
from dataclasses import dataclass
import json

from ..agents.codex import CodexAgent
from .learned_commands import LearnedCommands
from .session import Session


VOICE_REPLY_RULES = (
    'For the final reply, return a JSON object with a nonempty string "reply" and a boolean '
    '"listen_again". The "reply" field contains the text that will be displayed and spoken. '
    'Do not wrap the object in Markdown. Set "listen_again" to true only when you ask a direct '
    'question and need the user to answer to continue their request. Set it to false '
    'when the request is complete or the question is just an optional offer. '
    'If calling the desktop launch tool described above, use its one-line format instead.'
)


@dataclass(frozen=True)
class VoiceReply:
    text: str
    listen_again: bool = False


def _parse_voice_reply(response: str) -> VoiceReply:
    try:
        data = json.loads(response)
    except json.JSONDecodeError:
        return VoiceReply(response)
    if isinstance(data, dict) and ("reply" in data or "listen_again" in data):
        reply = data.get("reply")
        if isinstance(reply, str) and reply.strip():
            return VoiceReply(reply.strip(), listen_again=data.get("listen_again") is True)
        return VoiceReply("I couldn't read the reply.")
    return VoiceReply(response)


class Pipeline:
    def __init__(self, agent, max_history_turns: int = 8, *, max_history_chars: int = 4000):
        self.agent = agent
        self.session = Session(max_history_turns, max_history_chars=max_history_chars)

    def respond(
        self, text: str, *, spoken: bool = False,
        on_progress: Callable[[str], None] | None = None,
        learned_commands: LearnedCommands | None = None,
    ) -> str:
        return self._respond(
            text, spoken=spoken, on_progress=on_progress, learned_commands=learned_commands,
        ).text

    def respond_to_voice(
        self, text: str, *, allow_follow_up: bool, spoken: bool = False,
        on_progress: Callable[[str], None] | None = None,
        learned_commands: LearnedCommands | None = None,
        check_interruption: Callable[[], None] | None = None,
    ) -> VoiceReply:
        return self._respond(
            text, spoken=spoken, on_progress=on_progress, learned_commands=learned_commands,
            allow_follow_up=allow_follow_up, check_interruption=check_interruption,
        )

    def _respond(
        self, text: str, *, spoken: bool,
        on_progress: Callable[[str], None] | None,
        learned_commands: LearnedCommands | None,
        allow_follow_up: bool = False,
        check_interruption: Callable[[], None] | None = None,
    ) -> VoiceReply:
        text = text.strip()
        if not text:
            raise ValueError("Message cannot be empty")
        prompt = self.session.prompt(text, spoken=spoken)
        if allow_follow_up:
            prompt = f"{VOICE_REPLY_RULES}\n\n{prompt}"
        candidates = learned_commands.candidates(text) if learned_commands is not None else {}
        if candidates:
            prompt = f"{learned_commands.instructions(candidates)}\n\n{prompt}"
        options = {}
        if check_interruption is not None:
            options["check_interruption"] = check_interruption
        if on_progress is not None and isinstance(self.agent, CodexAgent):
            options["on_progress"] = on_progress
        response = self.agent.ask(prompt, **options)
        if check_interruption is not None:
            check_interruption()
        if candidates:
            response = learned_commands.finish(
                text, response, candidates, self.agent.cwd, self.agent.timeout_seconds,
            )
        reply = _parse_voice_reply(response) if allow_follow_up else VoiceReply(response)
        self.session.remember(text, reply.text)
        return reply
