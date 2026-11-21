"""Locally configured actions matched against complete voice transcripts."""
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys


def normalize_activation(phrase: str) -> str:
    return " ".join(phrase.strip().casefold().strip(".,!?;:¿¡").split())


class CommandError(RuntimeError):
    """A configured command was matched but could not complete."""


@dataclass(frozen=True)
class LocalCommand:
    argv: tuple[str, ...]
    reply: str

    def run(self, directory: Path, timeout: int) -> None:
        try:
            result = subprocess.run(
                self.argv, cwd=directory, stdin=subprocess.DEVNULL,
                capture_output=True, text=True, errors="replace", timeout=timeout,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise CommandError(f"Local command failed: {exc}") from exc
        if result.returncode:
            detail = (result.stderr or result.stdout).strip()[-500:]
            raise CommandError(f"Local command exited {result.returncode}. {detail}".rstrip())


def load_commands(entries: object) -> dict[str, LocalCommand]:
    if not isinstance(entries, list):
        raise ValueError("Use [[commands]] for each local command")
    commands = {}
    interpreters = {"bash": "/usr/bin/bash", "python": sys.executable}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Each commands entry must be a TOML table")
        activation = entry.get("activation")
        if "aliases" in entry:
            raise ValueError("Move commands.aliases into the commands.activation list")
        phrases = [activation] if isinstance(activation, str) else activation
        if not isinstance(phrases, list) or not phrases:
            raise ValueError("commands.activation must be a phrase or a non-empty list of phrases")
        code = entry.get("code")
        reply = entry.get("reply", "Running the command.")
        if not isinstance(code, str):
            raise ValueError(f"Set commands.code for {activation}")
        language, _, source = code.partition(":")
        interpreter = interpreters.get(language.strip())
        if not interpreter or not source.strip():
            raise ValueError(f"commands.code for {activation} must start with 'bash:' or 'python:'")
        if not isinstance(reply, str) or not reply.strip():
            raise ValueError(f"commands.reply for {activation} must be a non-empty string")
        command = LocalCommand((interpreter, "-c", source.lstrip()), reply)
        for phrase in phrases:
            if not isinstance(phrase, str) or not (key := normalize_activation(phrase)):
                raise ValueError("commands.activation must contain only non-empty phrases")
            if key in commands:
                raise ValueError(f"Duplicate command phrase: {phrase}")
            commands[key] = command
    return commands
