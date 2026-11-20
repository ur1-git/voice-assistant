"""Transcribe a WAV locally using whisper.cpp's command-line executable."""
from pathlib import Path
import subprocess


class TranscriptionError(RuntimeError):
    """Whisper failed to return a transcript."""


def transcribe(
    audio: Path, *, model: Path, language: str, timeout_seconds: int,
) -> str:
    output = audio.with_suffix(".txt")
    command = [
        "whisper-cli", "--model", str(model), "--file", str(audio),
        "--language", language, "--output-txt", "--output-file", str(output.with_suffix("")),
    ]
    try:
        result = subprocess.run(
            command, stdin=subprocess.DEVNULL, capture_output=True,
            timeout=timeout_seconds, check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise TranscriptionError(f"Whisper exceeded its {timeout_seconds}s timeout") from exc

    if result.returncode:
        detail = (result.stderr or result.stdout).decode(errors="replace").strip()
        raise TranscriptionError(f"whisper-cli failed (exit {result.returncode}): {detail[-1200:]}")
    try:
        return output.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError) as exc:
        raise TranscriptionError("whisper-cli did not produce a readable text transcript") from exc
