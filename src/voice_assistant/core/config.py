"""Configuration loaded from a local TOML file."""
from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class AgentConfig:
    provider: str = "codex"
    working_directory: Path = Path(".")
    max_history_turns: int = 8
    timeout_seconds: int = 180


@dataclass(frozen=True)
class KokoroConfig:
    model: Path
    voices: Path
    voice: str
    language: str


@dataclass(frozen=True)
class VoiceConfig:
    agent: AgentConfig
    device: str
    max_record_seconds: int
    model: Path
    language: str
    transcription_timeout: int
    kokoro: KokoroConfig | None = None


def _read_config(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _table(raw: dict, name: str) -> dict:
    data = raw.get(name, {})
    if not isinstance(data, dict):
        raise ValueError(f"{name} must be a TOML table")
    return data


def _resolve_path(value: str, config_path: Path) -> Path:
    path = Path(value).expanduser()
    return (config_path.parent / path).resolve()


def load_config(path: Path) -> AgentConfig:
    path = path.expanduser().resolve()
    return _agent_config(_read_config(path), path)


def _agent_config(raw: dict, path: Path) -> AgentConfig:
    data = _table(raw, "agent")
    provider = str(data.get("provider", "codex"))
    if provider not in ("codex", "claude"):
        raise ValueError("agent.provider must be 'codex' or 'claude'")
    directory = _resolve_path(str(data.get("working_directory", ".")), path)
    if not directory.is_dir():
        raise ValueError(f"Agent working directory not found: {directory}")
    turns = int(data.get("max_history_turns", 8))
    timeout = int(data.get("timeout_seconds", 180))
    if turns < 0 or timeout < 1:
        raise ValueError("History must be >= 0 and timeout >= 1")
    return AgentConfig(provider, directory, turns, timeout)


def load_voice_config(path: Path, *, speak: bool = False) -> VoiceConfig:
    path = path.expanduser().resolve()
    raw = _read_config(path)
    audio = _table(raw, "audio")
    stt = _table(raw, "stt")
    device = audio.get("device", "auto")
    duration = audio.get("max_record_seconds", 30)
    model = stt.get("model", "models/ggml-small.bin")
    language = stt.get("language", "auto")
    timeout = stt.get("timeout_seconds", 90)

    if stt.get("backend", "whisper_cpp") != "whisper_cpp":
        raise ValueError("stt.backend must be 'whisper_cpp'")
    for name, value in (("audio.device", device), ("stt.model", model), ("stt.language", language)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")
    for name, value in (("audio.max_record_seconds", duration), ("stt.timeout_seconds", timeout)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")

    kokoro = None
    if speak:
        tts = _table(raw, "tts")
        model_path = tts.get("model", "models/kokoro-v1.0.onnx")
        voices_path = tts.get("voices", "models/voices-v1.0.bin")
        voice = tts.get("voice", "af_heart")
        speech_language = tts.get("language", "en-us")
        for name, value in (
            ("model", model_path), ("voices", voices_path),
            ("voice", voice), ("language", speech_language),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"tts.{name} must be a non-empty string")
        kokoro = KokoroConfig(
            _resolve_path(model_path, path), _resolve_path(voices_path, path),
            voice, speech_language,
        )

    return VoiceConfig(
        agent=_agent_config(raw, path),
        device=device,
        max_record_seconds=duration,
        model=_resolve_path(model, path),
        language=language,
        transcription_timeout=timeout,
        kokoro=kokoro,
    )
