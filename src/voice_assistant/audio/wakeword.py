"""Listen for a local openWakeWord model using PipeWire PCM capture."""
from array import array
import math
from pathlib import Path
import subprocess
import wave

from .pipewire import RecordingError


class WakeWordDetector:
    def __init__(self, model: Path, threshold: float):
        paths = (model, model.with_name("embedding_model.onnx"), model.with_name("melspectrogram.onnx"))
        for path in paths:
            if not path.is_file():
                raise ValueError(f"Wake-word model file not found: {path}")
        try:
            from numpy import frombuffer
            from openwakeword import Model
        except ImportError as exc:
            raise ValueError("Install the openWakeWord dependencies listed in the README.") from exc

        self.frombuffer = frombuffer
        self.threshold = threshold
        try:
            self.model = Model(
                wakeword_models=[str(model)],
                embedding_model_path=str(paths[1]),
                melspec_model_path=str(paths[2]),
                inference_framework="onnx",
            )
        except Exception as exc:
            raise ValueError(f"Could not initialize the wake-word model: {exc}") from exc

    def wait(self, device: str) -> None:
        self.model.reset()
        command = [
            "pw-record", f"--target={device}", "--rate=16000",
            "--channels=1", "--format=s16", "--raw", "-",
        ]
        with subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE) as recorder:
            try:
                while len(chunk := recorder.stdout.read(2560)) == 2560:
                    samples = self.frombuffer(chunk, dtype="<i2")
                    if max(self.model.predict(samples).values()) >= self.threshold:
                        return
                raise RecordingError("PipeWire stopped while listening for the wake word")
            finally:
                if recorder.poll() is None:
                    recorder.kill()


def play_chime(path: Path, *, waking: bool) -> None:
    rate = 16000
    samples = array("h")
    count = rate * 90 // 1000
    for frequency in ((660, 880) if waking else (880, 660)):
        for index in range(count):
            envelope = min(1, index / 120, (count - index) / 240)
            samples.append(round(5000 * envelope * math.sin(math.tau * frequency * index / rate)))
    with wave.open(str(path), "wb") as audio:
        audio.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        audio.writeframes(samples.tobytes())
    result = subprocess.run(["pw-play", str(path)], check=False)
    if result.returncode:
        raise RecordingError(f"Could not play the recording chime (pw-play exit {result.returncode})")
