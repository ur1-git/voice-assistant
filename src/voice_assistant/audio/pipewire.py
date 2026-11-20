"""Record a Whisper-compatible WAV using the desktop's PipeWire session."""
from pathlib import Path
import signal
import subprocess
import wave


class RecordingError(RuntimeError):
    """PipeWire failed to capture a usable recording."""


def record_wav(path: Path, *, device: str, max_seconds: int) -> None:
    command = [
        "pw-record", f"--target={device}", "--rate=16000",
        "--channels=1", "--format=s16", str(path),
    ]
    stop_requested = False
    # Keep terminal Ctrl+C in Python so PipeWire can close the WAV cleanly.
    with subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        start_new_session=True,
    ) as recorder:
        try:
            try:
                _, stderr = recorder.communicate(timeout=max_seconds)
            except (KeyboardInterrupt, subprocess.TimeoutExpired):
                stop_requested = recorder.poll() is None
                if stop_requested:
                    recorder.send_signal(signal.SIGINT)
                try:
                    _, stderr = recorder.communicate(timeout=3)
                except subprocess.TimeoutExpired as exc:
                    raise RecordingError("PipeWire did not stop recording") from exc
        finally:
            if recorder.poll() is None:
                recorder.kill()

    detail = stderr.decode(errors="replace").strip()
    # pw-record can finalize the WAV and exit 1 after SIGINT without an error message.
    stopped_cleanly = stop_requested and recorder.returncode == 1 and not detail
    if recorder.returncode and not stopped_cleanly:
        raise RecordingError(f"pw-record failed (exit {recorder.returncode}): {detail[-1200:]}")

    try:
        with wave.open(str(path), "rb") as recording:
            if not recording.readframes(1):
                raise RecordingError("No audio captured. Check the microphone and try again.")
    except (OSError, wave.Error, EOFError) as exc:
        raise RecordingError(f"PipeWire did not produce a readable WAV: {exc}") from exc
