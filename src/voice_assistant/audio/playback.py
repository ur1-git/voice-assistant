"""Listen for interruptions while the assistant is busy."""
from array import array
from collections import deque
from collections.abc import Callable
from contextlib import closing
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory
from threading import Event, Thread
from time import monotonic
from typing import Literal
import wave

from ..core.config import VoiceConfig
from ..stt.whisper_cpp import TranscriptionError, transcribe
from .end_of_speech import FRAME_SAMPLES, EndOfSpeechRecorder
from .pipewire import RecordingError, pcm_frames
from .wakeword import WAKE_FRAME_BYTES, WakeWordDetector


STOP_WORDS = r"\bstop[\W_]+now\b"
Interruption = Literal["wake", "stop"]


class VoiceInterrupted(Exception):
    def __init__(self, reason: Interruption):
        super().__init__(reason)
        self.reason = reason


def is_stop_request(transcript: str) -> bool:
    return bool(re.search(STOP_WORDS + r"[\W_]*\Z", transcript, re.IGNORECASE))


def _transcribed_stop(frames: deque[bytes], config: VoiceConfig, playing_text: set[str]) -> bool:
    try:
        with TemporaryDirectory(prefix="voice-assistant-stop-") as directory:
            path = Path(directory) / "interrupt.wav"
            with wave.open(str(path), "wb") as clip:
                clip.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                clip.writeframes(b"".join(frames))
            transcript = transcribe(
                path, model=config.model, language=config.language,
                timeout_seconds=min(5, config.transcription_timeout),
            )
    except (TranscriptionError, OSError) as exc:
        print(f"Could not transcribe interruption: {exc}", file=sys.stderr)
        return False

    if not is_stop_request(transcript):
        return False
    # Ignore a matching excerpt of the current playback that the microphone picked up.
    words = " ".join(re.findall(r"\w+", transcript.casefold()))
    return not any(
        f" {words} " in " " + " ".join(re.findall(r"\w+", message.casefold())) + " "
        for message in playing_text
    )


def listen_for_interruption(
    config: VoiceConfig, wake_word: WakeWordDetector | None, vad: EndOfSpeechRecorder | None,
    *, listening: Callable[[], bool], playing_text: Callable[[], str],
) -> Interruption | None:
    if wake_word is not None:
        wake_word.reset()
    if vad is not None:
        vad.reset()
    vad_pending = bytearray()

    def is_voiced(frame: bytes) -> bool:
        if vad is None:
            samples = array("h", frame)
            return sum(sample * sample for sample in samples) >= len(samples) * 120**2
        vad_pending.extend(frame)
        voiced = False
        while len(vad_pending) >= FRAME_SAMPLES * 2:
            segment = bytes(vad_pending[:FRAME_SAMPLES * 2])
            del vad_pending[:FRAME_SAMPLES * 2]
            voiced |= vad.speech_probability(segment) >= 0.3
        return voiced

    wake_at = None
    lead_in: deque[bytes] = deque(maxlen=3)
    clip: deque[bytes] = deque(maxlen=64)
    playback_messages: set[str] = set()
    voiced_frames = 0
    quiet_frames = 0
    with closing(pcm_frames(config.device, WAKE_FRAME_BYTES, listening=listening)) as frames:
        for frame in frames:
            if wake_word is not None and wake_at is None and wake_word.detected(frame):
                wake_at = monotonic()

            voiced = is_voiced(frame)
            if voiced:
                if not clip:
                    clip.extend(lead_in)
                voiced_frames += 1
                quiet_frames = 0
            elif clip:
                quiet_frames += 1
            if clip or voiced:
                clip.append(frame)
                if message := playing_text():
                    playback_messages.add(message)
            lead_in.append(frame)

            # A single utterance can contain both the wake word and “stop now”.
            if quiet_frames >= 4 or (wake_at is not None and monotonic() - wake_at >= 1.2):
                if voiced_frames >= 2 and _transcribed_stop(clip, config, playback_messages):
                    return "stop"
                if wake_at is not None:
                    return "wake"
                clip.clear()
                playback_messages.clear()
                voiced_frames = quiet_frames = 0

    if voiced_frames >= 2 and _transcribed_stop(clip, config, playback_messages):
        return "stop"
    return "wake" if wake_at is not None else None


class InterruptionListener:
    """Own one microphone stream across the agent call and all of its speech."""

    def __init__(self, config: VoiceConfig, wake_word: WakeWordDetector | None, vad: EndOfSpeechRecorder | None):
        self.config = config
        self.wake_word = wake_word
        self.vad = vad
        self.result: Interruption | None = None
        self.playing_text = ""
        self._close = Event()
        self._interrupted = Event()
        self._thread = Thread(target=self._listen, name="voice-interruptions")

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._close.set()
        self._thread.join()

    def _listen(self) -> None:
        try:
            self.result = listen_for_interruption(
                self.config, self.wake_word, self.vad,
                listening=lambda: not self._close.is_set(), playing_text=lambda: self.playing_text,
            )
        except (RecordingError, OSError) as exc:
            print(f"Voice interruption unavailable: {exc}", file=sys.stderr)
        if self.result is not None:
            self._interrupted.set()

    def check(self) -> None:
        if self.result is not None:
            raise VoiceInterrupted(self.result)

    def stop_playback(self, message: str, playing: Callable[[], bool]) -> bool:
        self.playing_text = message
        try:
            while playing() and not self._interrupted.wait(0.05):
                pass
            return self._interrupted.is_set()
        finally:
            self.playing_text = ""
