"""Configuration loaded from a local TOML file."""
from dataclasses import dataclass, field
from pathlib import Path
import tomllib

from .commands import LocalCommand, load_commands


@dataclass(frozen=True)
class AgentConfig:
    provider: str = "codex"
    working_directory: Path = Path(".")
    max_history_turns: int = 8
    timeout_seconds: int = 180
    max_history_chars: int = 4000
    codex_full_access: bool = False
    codex_model: str = "gpt-6.2-sol"
    codex_reasoning_effort: str = "low"


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
    wake_model: Path | None = None
    wake_threshold: float = 0.5
    end_silence_ms: int = 1000
    commands: dict[str, LocalCommand] = field(default_factory=dict)


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
    history_chars = data.get("max_history_chars", 4000)
    if turns < 0 or timeout < 1:
        raise ValueError("History must be >= 0 and timeout >= 1")
    if type(history_chars) is not int or history_chars < 0:
        raise ValueError("agent.max_history_chars must be a nonnegative integer")
    codex_full_access = data.get("codex_full_access", False)
    if type(codex_full_access) is not bool:
        raise ValueError("agent.codex_full_access must be true or false")
    codex_model = data.get("codex_model", AgentConfig.codex_model)
    codex_reasoning_effort = data.get("codex_reasoning_effort", AgentConfig.codex_reasoning_effort)
    for name, value in (
        ("codex_model", codex_model),
        ("codex_reasoning_effort", codex_reasoning_effort),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"agent.{name} must be a non-empty string")
    return AgentConfig(
        provider, directory, turns, timeout, history_chars, codex_full_access,
        codex_model, codex_reasoning_effort,
    )


def load_voice_config(path: Path, *, speak: bool = False, wake: bool = False) -> VoiceConfig:
    path = path.expanduser().resolve()
    raw = _read_config(path)
    audio = _table(raw, "audio")
    stt = _table(raw, "stt")
    device = audio.get("device", "auto")
    duration = audio.get("max_record_seconds", 30)
    model = stt.get("model", "models/ggml-small.bin")
    language = stt.get("language", "en")
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

    wake_model = None
    wake_threshold = 0.5
    end_silence_ms = 1000
    if wake:
        wakeword = _table(raw, "wakeword")
        model_path = wakeword.get("model", "models/openwakeword/hey_jarvis_v0.1.onnx")
        wake_threshold = wakeword.get("threshold", 0.5)
        if not isinstance(model_path, str) or not model_path.strip():
            raise ValueError("wakeword.model must be a non-empty string")
        if type(wake_threshold) not in (int, float) or not 0 < wake_threshold <= 1:
            raise ValueError("wakeword.threshold must be greater than 0 and at most 1")
        wake_model = _resolve_path(model_path, path)
        end_silence_ms = audio.get("end_silence_ms", 1000)
        if type(end_silence_ms) is not int or end_silence_ms < 1:
            raise ValueError("audio.end_silence_ms must be a positive integer")

    return VoiceConfig(
        agent=_agent_config(raw, path),
        device=device,
        max_record_seconds=duration,
        model=_resolve_path(model, path),
        language=language,
        transcription_timeout=timeout,
        kokoro=kokoro,
        wake_model=wake_model,
        wake_threshold=wake_threshold,
        end_silence_ms=end_silence_ms,
        commands=load_commands(raw.get("commands", [])),
    )
