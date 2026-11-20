# Linux voice assistant

A modular assistant for the Linux desktop. Use Codex or Claude Code for project questions and follow-ups while keeping your editor in focus. The terminal supports typed requests and key-triggered microphone input. The planned hands-free loop will wake on a keyword and read the answer aloud.

## Portable Linux bundle

Extract `voice-assistant-linux-x86_64.tar.gz` and open a terminal in the extracted folder. Codex is the default AI backend and ships with the bundle, alongside Python and CPU-based whisper.cpp with the multilingual Whisper base model. Voice mode uses your desktop's PipeWire session and needs `pw-record`, either supplied with the bundle or already available on the desktop.

```bash
./voice-assistant doctor
./voice-assistant login
./voice-assistant voice
```

Sign in with your own account. The assistant uses the extracted folder's `config.toml`. Set `agent.working_directory` there to point to your project. The default is the bundle's empty `workspace/` directory.

To use Claude Code, run `./voice-assistant login claude` to download it into the folder and sign in, then run `./voice-assistant voice --agent claude`. The agent service needs an internet connection. Whisper transcription runs locally.

The launcher directs private configuration, credentials, caches, and temporary files into `.state/` inside the folder. To remove the bundle, close the app and delete the extracted folder. Share the original archive so each recipient starts with a fresh copy.

### Build the bundle

On an x86-64 Linux build machine with Python 3.12+, CMake, a C/C++ toolchain, and internet access, run:

```bash
python portable/build.py
```

The builder downloads upstream releases into `dist/.downloads/` and creates `dist/voice-assistant-linux-x86_64.tar.gz`. You can also supply cached downloads with `--assets /path/to/downloads`. To include the recording client, pass `--capture /path/to/recording-prefix` with a redistributable `bin/pw-record` and any client libraries and license files it needs. Build on the oldest Linux release you plan to support, since the native Whisper executable depends on the build machine's Linux C library version.

## Development setup

Use Python 3.11 or later and install and authenticate either `codex` or `claude`. The assistant has no Python runtime dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
cp config.example.toml config.toml
voice-assistant doctor
voice-assistant chat --agent codex
```

Set `agent.provider` to `codex` or `claude` and `agent.working_directory` to the project you want to work with. Relative paths resolve from the TOML file. The `--agent` flag overrides the configured provider. Use `--config /path/to/config.toml` before the subcommand to select a different configuration.

For a single typed request:

```bash
voice-assistant ask --agent claude "Explain what this project does"
```

## Microphone input

Use a running PipeWire desktop session with `pw-record`, and put `whisper-cli` on your `PATH`. Follow the upstream build and model instructions to set up whisper.cpp. Set `stt.model` in `config.toml` to the downloaded GGML model file. A multilingual model such as `ggml-small.bin` supports language detection with `stt.language = "auto"`. You can also set a language code such as `"en"` or `"es"`. [whisper.cpp build and model instructions](https://github.com/ggml-org/whisper.cpp#quick-start)

```bash
voice-assistant voice
```

Press Enter to record, speak your request, then press Ctrl+C to stop recording. Capture also stops at `audio.max_record_seconds` (30 seconds by default). The terminal shows the transcript and the agent’s reply. Press Enter again for a follow-up in the same session or type `/quit` to exit.

The default PipeWire input is used automatically. To choose another microphone, run `wpctl status --name` and set `audio.device` to its node name under Sources. The app writes each recording as a temporary 16 kHz mono WAV for local Whisper transcription and removes the temporary files after processing. The selected AI backend receives the transcript and the bounded conversation history. Codex can also read project files to answer questions.

## Milestones

1. **Text prototype.** Send a typed question and follow-up to either backend, selected through TOML or the CLI.
2. **Microphone input.** Trigger a recording from the terminal and see the local Whisper transcript and the agent’s answer. Evaluate Vulkan acceleration for Whisper on the Radeon RX 9070 XT.
3. **Spoken replies.** Send the final answer to Kokoro for local speech synthesis and play it through PipeWire on the selected output. Support playback interruption and return to readiness for another request.
4. **Hands-free loop.** Use openWakeWord to start a turn and Silero VAD to end recording after a pause. Resume wake-word listening after playback to avoid triggering on the assistant’s voice. Detection and synthesis run locally.
5. **Desktop service.** Start at login through a systemd user service and support starting, stopping, and restarting it as a regular user. Persistent sessions are planned separately from the current in-memory history.

## Agent access

The first release covers questions and read-only project inspection. Codex runs with `--sandbox read-only`, and Claude Code runs with its tools disabled. The CLI processes run as your Linux user.

Approved project changes are later work. Before enabling write actions, define which actions require approval and enforce the rule in the execution layer. For a destructive or risky action, show the exact change and target and require two explicit confirmations. A denial or missing confirmation cancels the action. The assistant should never have sudo access.

## Layout and tests

- `src/voice_assistant/agents/` runs the agent CLIs and extracts their final replies.
- `src/voice_assistant/core/` loads TOML settings and coordinates conversation history.
- `src/voice_assistant/audio/` records from PipeWire.
- `src/voice_assistant/stt/` transcribes through whisper.cpp.
- `src/voice_assistant/tts/` is reserved for the Kokoro adapter.
- `tests/` contains offline checks.

```bash
python -m unittest discover -s tests -v
```
