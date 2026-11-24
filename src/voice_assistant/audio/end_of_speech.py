"""Record a request until Silero detects a pause after speech."""
import os
from pathlib import Path
from selectors import DefaultSelector, EVENT_READ
from time import monotonic
import wave

from .pipewire import RecordingError, pcm_capture


FRAME_SAMPLES = 480
FRAME_MS = 30
START_TIMEOUT_SECONDS = 5


class EndOfSpeechRecorder:
    def __init__(self, model_path: Path, silence_ms: int):
        if not model_path.is_file():
            raise ValueError(f"Silero VAD model not found: {model_path}. See the wake phrase setup in the README.")
        try:
            from numpy import frombuffer
            from openwakeword.vad import VAD

            self.model = VAD(str(model_path))
        except Exception as exc:
            raise ValueError(f"Could not initialize Silero VAD: {exc}") from exc
        self.frombuffer = frombuffer
        self.silence_ms = silence_ms

    def record(
        self, path: Path, *, device: str, max_seconds: int,
        start_timeout_seconds: int = START_TIMEOUT_SECONDS,
    ) -> bool:
        self.model.reset_states()
        speech_started = False
        voiced_frames = 0
        quiet_ms = 0
        pending = bytearray()
        started_at = monotonic()
        deadline = started_at + max_seconds
        start_deadline = min(deadline, started_at + start_timeout_seconds)
        with wave.open(str(path), "wb") as audio:
            audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            try:
                with pcm_capture(device) as stream, DefaultSelector() as selector:
                    selector.register(stream, EVENT_READ)
                    while True:
                        cutoff = deadline if speech_started else start_deadline
                        remaining = cutoff - monotonic()
                        if remaining <= 0 or not selector.select(remaining):
                            return speech_started
                        chunk = os.read(stream.fileno(), FRAME_SAMPLES * 2 - len(pending))
                        if not chunk:
                            raise RecordingError("PipeWire stopped during request capture")
                        pending.extend(chunk)
                        if len(pending) < FRAME_SAMPLES * 2:
                            continue
                        frame = bytes(pending)
                        pending.clear()
                        audio.writeframesraw(frame)
                        samples = self.frombuffer(frame, dtype="<i2")
                        if self.model.predict(samples, frame_size=FRAME_SAMPLES) >= 0.5:
                            voiced_frames += 1
                            speech_started |= voiced_frames >= 2
                            quiet_ms = 0
                        else:
                            voiced_frames = 0
                            quiet_ms += FRAME_MS
                            if speech_started and quiet_ms >= self.silence_ms:
                                return True
            except KeyboardInterrupt:
                return audio.getnframes() > 0
