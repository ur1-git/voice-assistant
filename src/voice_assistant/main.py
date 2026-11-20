"""Terminal entry points for typed and spoken requests."""
import argparse
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory

from .agents.base import AgentError
from .agents.claude import ClaudeAgent
from .agents.codex import CodexAgent
from .audio.pipewire import RecordingError, record_wav
from .core.config import VoiceConfig, load_config, load_voice_config
from .core.pipeline import Pipeline
from .stt.whisper_cpp import TranscriptionError, transcribe
from .tts.kokoro import KokoroSpeaker, SpeechError


def voice_chat(pipeline: Pipeline, config: VoiceConfig) -> None:
    speaker = KokoroSpeaker(config.kokoro) if config.kokoro is not None else None
    while True:
        try:
            choice = input("Press Enter to record or type /quit to exit. ").strip()
            if choice in ("/quit", "/exit"):
                return
            if choice:
                continue

            with TemporaryDirectory(prefix="voice-assistant-") as directory:
                audio = Path(directory) / "request.wav"
                print(f"Recording for up to {config.max_record_seconds}s. Press Ctrl+C to stop.", flush=True)
                record_wav(audio, device=config.device, max_seconds=config.max_record_seconds)
                print("\nTranscribing…", flush=True)
                transcript = transcribe(
                    audio, model=config.model, language=config.language,
                    timeout_seconds=config.transcription_timeout,
                )

            if not transcript or transcript == "[BLANK_AUDIO]":
                print("No speech transcribed.\n")
                continue
            print(f"Transcript> {transcript}", flush=True)
            reply = pipeline.respond(transcript, spoken=speaker is not None)
            print("Assistant> " + reply + "\n", flush=True)
            if speaker:
                print("Speaking… Press Ctrl+C to stop.", flush=True)
                try:
                    speaker.speak(reply)
                except KeyboardInterrupt:
                    print("\nSpeech stopped.")
        except (EOFError, KeyboardInterrupt):
            print()
            return
        except (RecordingError, TranscriptionError, AgentError, SpeechError, OSError, ValueError) as exc:
            print(f"Error: {exc}", file=sys.stderr)


def main(argv=None, *, agent_executables: dict[str, str] | None = None) -> int:
    executables = agent_executables or {}
    parser = argparse.ArgumentParser(prog="voice-assistant")
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Check local tool availability")
    for name, help_text in (
        ("ask", "Send one typed request"),
        ("chat", "Start a text session"),
        ("voice", "Record and transcribe spoken requests"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--agent", choices=("codex", "claude"))
        if name == "voice":
            command.add_argument("--speak", action="store_true", help="Read replies aloud with Kokoro")
        if name == "ask":
            command.add_argument("message")
    args = parser.parse_args(argv)

    if args.command == "doctor":
        print(f"Python: {sys.version.split()[0]}")
        for tool in ("codex", "claude", "pactl", "pw-record", "pw-play", "whisper-cli", "vulkaninfo"):
            print(f"{tool}: {shutil.which(executables.get(tool, tool)) or 'not found'}")
        return 0

    try:
        voice_config = load_voice_config(args.config, speak=args.speak) if args.command == "voice" else None
        if voice_config is not None:
            tools = ("pw-record", "whisper-cli")
            if voice_config.kokoro is not None:
                tools += ("pw-play",)
            for tool in tools:
                if not shutil.which(tool):
                    raise ValueError(f"Executable '{tool}' not found in PATH")
            if not voice_config.model.is_file():
                raise ValueError(f"Whisper model not found: {voice_config.model}. Set stt.model in the config.")
        config = voice_config.agent if voice_config is not None else load_config(args.config)
        provider = args.agent or config.provider
        agent_cls = CodexAgent if provider == "codex" else ClaudeAgent
        agent = agent_cls(executables.get(provider, provider), config.working_directory, config.timeout_seconds)
        agent.ensure_available()
        pipeline = Pipeline(agent, config.max_history_turns)
        if voice_config is not None:
            voice_chat(pipeline, voice_config)
        elif args.command == "ask":
            print(pipeline.respond(args.message))
        else:
            print(f"Chatting with {provider}. Type /quit to exit.")
            while True:
                try:
                    question = input("You> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                if question in ("/quit", "/exit"):
                    break
                if not question:
                    continue
                try:
                    print("Assistant> " + pipeline.respond(question) + "\n")
                except (AgentError, ValueError) as exc:
                    print(f"Error: {exc}", file=sys.stderr)
    except (AgentError, SpeechError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
