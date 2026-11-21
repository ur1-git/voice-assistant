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
    capture_output: bool = True
    run_before_reply: bool = True

    def run(self, directory: Path, timeout: int) -> None:
        output = subprocess.PIPE if self.capture_output else subprocess.DEVNULL
        try:
            result = subprocess.run(
                self.argv, cwd=directory, stdin=subprocess.DEVNULL,
                stdout=output, stderr=output, text=True, errors="replace", timeout=timeout,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise CommandError(f"Local command failed: {exc}") from exc
        if result.returncode:
            detail = (result.stderr or result.stdout or "").strip()[-500:]
            raise CommandError(f"Local command exited {result.returncode}. {detail}".rstrip())


def load_commands(entries: object) -> dict[str, LocalCommand]:
    if not isinstance(entries, list):
        raise ValueError("Use [[commands]] for each local command")
    commands = {}
    interpreters = {"bash": "/usr/bin/bash", "python": sys.executable}
    code_fields = {"codeBash": "bash", "codePython": "python"}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Each commands entry must be a TOML table")
        activation = entry.get("activation")
        if "aliases" in entry:
            raise ValueError("Move commands.aliases into the commands.activation list")
        phrases = [activation] if isinstance(activation, str) else activation
        if not isinstance(phrases, list) or not phrases:
            raise ValueError("commands.activation must be a phrase or a non-empty list of phrases")
        fields = [field for field in (*code_fields, "code") if field in entry]
        if len(fields) != 1:
            raise ValueError(f"Use exactly one of codeBash or codePython for {activation}")
        field = fields[0]
        source = entry[field]
        if not isinstance(source, str) or not source.strip():
            raise ValueError(f"commands.{field} for {activation} must be a non-empty string")
        if field == "code":
            language, _, source = source.partition(":")
            language = language.strip()
            if language not in interpreters or not source.strip():
                raise ValueError(f"Use codeBash or codePython for {activation}, or prefix legacy code with 'bash:' or 'python:'")
        else:
            language = code_fields[field]
        run_before_reply = entry.get("runBeforeReply", True)
        if type(run_before_reply) is not bool:
            raise ValueError(f"commands.runBeforeReply for {activation} must be true or false")
        reply = entry.get("reply", "Done." if run_before_reply else "Running the command.")
        if not isinstance(reply, str) or not reply.strip():
            raise ValueError(f"commands.reply for {activation} must be a non-empty string")
        command = LocalCommand(
            (interpreters[language], "-c", source.lstrip()), reply, run_before_reply=run_before_reply,
        )
        for phrase in phrases:
            if not isinstance(phrase, str) or not (key := normalize_activation(phrase)):
                raise ValueError("commands.activation must contain only non-empty phrases")
            if key in commands:
                raise ValueError(f"Duplicate command phrase: {phrase}")
            commands[key] = command
    return commands
