"""Synthesize replies locally and play them through PipeWire."""
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

from ..core.config import KokoroConfig


class SpeechError(RuntimeError):
    """Kokoro synthesis or playback could not complete."""


class KokoroSpeaker:
    def __init__(self, config: KokoroConfig):
        for path in (config.model, config.voices):
            if not path.is_file():
                raise SpeechError(f"Kokoro asset not found: {path}")
        try:
            from kokoro_onnx import Kokoro
            from onnxruntime import InferenceSession
            from soundfile import write
        except ImportError as exc:
            raise SpeechError("Install the Kokoro dependencies with pip install -e '.[tts]'") from exc

        class KokoroSession(InferenceSession):
            def __init__(self, model: Path):
                super().__init__(str(model), providers=["CPUExecutionProvider"])
                self.float_speed = any(
                    item.name == "speed" and item.type == "tensor(float)"
                    for item in self.get_inputs()
                )

            def run(self, output_names, input_feed, run_options=None):
                # Kokoro ONNX 0.4.7 sends int32 speed to some float-speed exports.
                if self.float_speed:
                    input_feed = {**input_feed, "speed": input_feed["speed"].astype("float32")}
                return super().run(output_names, input_feed, run_options)

        self.config = config
        self.write = write
        try:
            self.engine = Kokoro.from_session(KokoroSession(config.model), str(config.voices))
            if config.voice not in self.engine.get_voices():
                raise ValueError(f"Unknown Kokoro voice: {config.voice}")
        except Exception as exc:
            raise SpeechError(f"Could not initialize Kokoro: {exc}") from exc

    def synthesize(self, text: str, path: Path) -> None:
        try:
            samples, rate = self.engine.create(
                text, voice=self.config.voice, lang=self.config.language,
            )
            self.write(path, samples, rate, subtype="PCM_16")
        except Exception as exc:
            raise SpeechError(f"Kokoro synthesis failed: {exc}") from exc

    def speak(self, text: str) -> None:
        if not text.strip():
            return
        with TemporaryDirectory(prefix="voice-assistant-tts-") as directory:
            output = Path(directory) / "reply.wav"
            self.synthesize(text, output)
            result = subprocess.run(["pw-play", str(output)], check=False)
            if result.returncode:
                raise SpeechError(f"pw-play failed (exit {result.returncode})")
