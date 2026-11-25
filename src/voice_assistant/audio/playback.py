"""Recognize local interruptions while a reply is playing."""
from array import array
from collections import deque
from collections.abc import Callable
from contextlib import closing
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory
from time import monotonic
from typing import Literal
import wave

from ..core.config import VoiceConfig
from ..stt.whisper_cpp import TranscriptionError, transcribe
from .pipewire import pcm_frames
from .wakeword import WAKE_FRAME_BYTES, WakeWordDetector


STOP_WORDS = r"\bstop[\W_]+now\b"
Interruption = Literal["wake", "stop"]


def is_stop_request(transcript: str) -> bool:
    return bool(re.search(STOP_WORDS + r"[\W_]*\Z", transcript, re.IGNORECASE))


def _transcribed_stop(frames: deque[bytes], config: VoiceConfig) -> bool:
    try:
        with TemporaryDirectory(prefix="voice-assistant-stop-") as directory:
            path = Path(directory) / "interrupt.wav"
            with wave.open(str(path), "wb") as clip:
                clip.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                clip.writeframes(b"".join(frames))
            return is_stop_request(transcribe(
                path, model=config.model, language=config.language,
                timeout_seconds=min(5, config.transcription_timeout),
            ))
    except (TranscriptionError, OSError) as exc:
        print(f"Could not transcribe playback interruption: {exc}", file=sys.stderr)
        return False


def listen_for_interruption(
    message: str, config: VoiceConfig, wake_word: WakeWordDetector | None,
    *, playing: Callable[[], bool],
) -> Interruption | None:
    # Recognizing the same words from the microphone could pick up the reply itself.
    if re.search(STOP_WORDS, message, re.IGNORECASE):
        return "wake" if wake_word and wake_word.wait(config.device, listening=playing) else None

    if wake_word is not None:
        wake_word.reset()
    wake_at = None
    playback_ended_at = None
    lead_in: deque[bytes] = deque(maxlen=3)
    clip: deque[bytes] = deque(maxlen=64)  # Only the last five seconds are needed for the suffix.
    voiced_frames = 0
    quiet_frames = 0

    def listening() -> bool:
        nonlocal playback_ended_at
        if playing():
            return True
        if not voiced_frames:
            return False
        if playback_ended_at is None:
            playback_ended_at = monotonic()
        return monotonic() - playback_ended_at < 0.7

    with closing(pcm_frames(config.device, WAKE_FRAME_BYTES, listening=listening)) as frames:
        for frame in frames:
            if wake_word is not None and wake_at is None and wake_word.detected(frame):
                wake_at = monotonic()

            samples = array("h", frame)
            loud_enough = sum(sample * sample for sample in samples) >= len(samples) * 200**2
            if loud_enough:
                if not clip:
                    clip.extend(lead_in)
                voiced_frames += 1
                quiet_frames = 0
            elif clip:
                quiet_frames += 1
            if clip or loud_enough:
                clip.append(frame)
            lead_in.append(frame)

            # Give a wake phrase in the same utterance time to be followed by “stop now”.
            if quiet_frames >= 4 or (wake_at is not None and monotonic() - wake_at >= 1.2):
                if voiced_frames >= 2 and _transcribed_stop(clip, config):
                    return "stop"
                if wake_at is not None:
                    return "wake"
                clip.clear()
                voiced_frames = quiet_frames = 0

    if voiced_frames >= 2 and _transcribed_stop(clip, config):
        return "stop"
    return "wake" if wake_at is not None else None
