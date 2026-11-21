"""Terminal entry points for typed and spoken requests."""
import argparse
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory

from .agents.base import AgentError
from .agents.claude import ClaudeAgent
from .agents.codex import CodexAgent
from .audio.end_of_speech import EndOfSpeechRecorder
from .audio.pipewire import RecordingError, record_wav
from .audio.wakeword import WakeWordDetector, play_chime
from .core.commands import CommandError, normalize_activation
from .core.config import VoiceConfig, load_config, load_voice_config
from .core.learned_commands import LearnedCommands
from .core.pipeline import Pipeline
from .stt.whisper_cpp import TranscriptionError, transcribe
from .tts.kokoro import KokoroSpeaker, SpeechError


def voice_chat(pipeline: Pipeline, config: VoiceConfig, learned_commands: LearnedCommands) -> None:
    speaker = KokoroSpeaker(config.kokoro) if config.kokoro is not None else None

    detector = (
        WakeWordDetector(config.wake_model, config.wake_threshold)
        if config.wake_model is not None else None
    )
    automatic_recorder = (
        EndOfSpeechRecorder(config.wake_model.with_name("silero_vad.onnx"), config.end_silence_ms)
        if config.wake_model is not None else None
    )

    def interrupt_playback(listening):
        try:
            return detector.wait(config.device, listening=listening)
        except (RecordingError, OSError) as exc:
            print(f"Playback interruption unavailable: {exc}", file=sys.stderr)
            return False

    def speak(message: str) -> bool:
        if speaker is None:
            return True
        try:
            if speaker.speak(message, interrupt=interrupt_playback if detector else None):
                return True
        except KeyboardInterrupt:
            pass
        print("Speech stopped.", flush=True)
        return False

    def announce_progress(message: str) -> None:
        print(f"Progress> {message}", flush=True)
        try:
            speak(message)
        except (SpeechError, OSError) as exc:
            print(f"Progress playback failed: {exc}", file=sys.stderr)

    while True:
        try:
            if detector:
                print("Listening for Hey Jarvis. Press Ctrl+C to quit.", flush=True)
                try:
                    detector.wait(config.device)
                except RecordingError as exc:
                    print(f"Error: {exc}", file=sys.stderr)
                    return
            else:
                choice = input("Press Enter to record or type /quit to exit. ").strip()
                if choice in ("/quit", "/exit"):
                    return
                if choice:
                    continue

            with TemporaryDirectory(prefix="voice-assistant-") as directory:
                audio = Path(directory) / "request.wav"
                chime = audio.with_name("chime.wav")
                if detector:
                    play_chime(chime, waking=True)
                if automatic_recorder:
                    print("Recording. Pause to send or press Ctrl+C to send now.", flush=True)
                    captured = automatic_recorder.record(
                        audio, device=config.device, max_seconds=config.max_record_seconds,
                    )
                    play_chime(chime, waking=False)
                    if not captured:
                        print("No speech detected.\n", flush=True)
                        continue
                else:
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
            command = config.commands.get(normalize_activation(transcript)) or learned_commands.lookup(transcript)
            if command is None:
                reply = pipeline.respond(
                    transcript, spoken=speaker is not None,
                    on_progress=announce_progress if speaker is not None else None,
                    learned_commands=learned_commands,
                )
            else:
                print("Assistant> " + command.reply + "\n", flush=True)
                if not speak(command.reply):
                    print("Local command canceled.", flush=True)
                    continue
                print("Running local command.", flush=True)
                try:
                    command.run(config.agent.working_directory, config.agent.timeout_seconds)
                except CommandError as exc:
                    print(f"Error: {exc}", file=sys.stderr)
                    print("Assistant> The local command failed.", flush=True)
                    speak("The local command failed.")
                continue
            print("Assistant> " + reply + "\n", flush=True)
            speak(reply)
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
            command.add_argument("--wake", action="store_true", help="Start recording when you say Hey Jarvis")
        if name == "ask":
            command.add_argument("message")
    args = parser.parse_args(argv)

    if args.command == "doctor":
        print(f"Python: {sys.version.split()[0]}")
        for tool in ("codex", "claude", "pactl", "pw-record", "pw-play", "whisper-cli", "gio", "vulkaninfo"):
            print(f"{tool}: {shutil.which(executables.get(tool, tool)) or 'not found'}")
        return 0

    try:
        voice_config = (
            load_voice_config(args.config, speak=args.speak, wake=args.wake)
            if args.command == "voice" else None
        )
        if voice_config is not None:
            tools = ("pw-record", "whisper-cli")
            if voice_config.kokoro is not None or voice_config.wake_model is not None:
                tools += ("pw-play",)
            for tool in tools:
                if not shutil.which(tool):
                    raise ValueError(f"Executable '{tool}' not found in PATH")
            if not voice_config.model.is_file():
                raise ValueError(f"Whisper model not found: {voice_config.model}. Set stt.model in the config.")
        config = voice_config.agent if voice_config is not None else load_config(args.config)
        provider = args.agent or config.provider
        executable = executables.get(provider, provider)
        if provider == "codex":
            agent = CodexAgent(
                executable, config.working_directory, config.timeout_seconds,
                full_access=config.codex_full_access,
                model=config.codex_model,
                reasoning_effort=config.codex_reasoning_effort,
            )
        else:
            agent = ClaudeAgent(executable, config.working_directory, config.timeout_seconds)
        if voice_config is None:
            agent.ensure_available()
        pipeline = Pipeline(agent, config.max_history_turns, max_history_chars=config.max_history_chars)
        if voice_config is not None:
            voice_chat(pipeline, voice_config, LearnedCommands(args.config))
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
