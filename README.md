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

## Spoken replies

In the development checkout, install the optional TTS dependencies in your project venv:

```bash
python -m pip install -e '.[tts]'
```

Place `kokoro-v1.0.onnx` and `voices-v1.0.bin` from the [Kokoro ONNX release](https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.1) in `models/`. The default voice is American English (`af_heart`). Set `tts.voice` and `tts.language` in your config to change the voice and pronunciation language. Kokoro synthesizes locally and uses `pw-play` on the default PipeWire output.

```bash
voice-assistant voice --speak
```

The app prints each reply before speaking it. Ctrl+C during synthesis or playback returns to the recording prompt. The portable bundle currently uses text replies.

With Codex and spoken replies enabled, Kokoro also reads short progress updates from the current request. The updates use local speech synthesis and stay out of the conversation history.

## Wake phrase

With the project venv active, install the ONNX dependencies and download the pretrained models:

```bash
python -m pip install numpy onnxruntime requests scipy scikit-learn tqdm
python -m pip install --no-deps openwakeword==0.6.0
mkdir -p models/openwakeword
for model in hey_jarvis_v0.1 embedding_model melspectrogram silero_vad; do
    curl -fL "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/$model.onnx" \
        -o "models/openwakeword/$model.onnx" || break
done
```

Run `./voice-assistant` from the project folder to start wake mode with spoken replies. The launcher uses the project's `.venv` and `config.toml`. Say “Hey Jarvis” and wait for the rising chime before speaking. Silero VAD sends the recording after about one second of quiet following speech. Press Ctrl+C during capture to send immediately. A falling chime marks the end of capture. If you never start speaking, the app returns to listening after five seconds without sending a request. After the reply, the app listens for the next wake phrase. Ctrl+C while waiting for the wake phrase exits the app.

The Silero ONNX model goes next to the wake model and uses the same ONNX Runtime already installed for openWakeWord. Set `audio.end_silence_ms` in `config.toml` to give yourself more or less pause time before sending. The existing `audio.max_record_seconds` still caps recordings.

The default detection threshold is 0.5. If you hear too many false wakes, raise `wakeword.threshold` in `config.toml`. If it misses your phrase, try lowering it and check for false wakes. Listening and detection run locally. OpenWakeWord 0.6.0 declares a TensorFlow Lite dependency on Linux, so the ONNX setup above installs its dependencies explicitly.

## Milestones

1. **Text prototype.** Send a typed question and follow-up to either backend, selected through TOML or the CLI.
2. **Microphone input.** Trigger a recording from the terminal and see the local Whisper transcript and the agent’s answer. Evaluate Vulkan acceleration for Whisper on the Radeon RX 9070 XT.
3. **Spoken replies.** Enable local Kokoro synthesis and PipeWire playback with `voice --speak`. Ctrl+C interrupts speech and returns to the recording prompt.
4. **Hands-free loop.** OpenWakeWord starts recording and Silero VAD sends the request when the user stops speaking. Chimes mark the start and end. Wake-word listening resumes after playback.
5. **Spoken stop command.** Recognize a dedicated stop word during reply playback and return to listening.
6. **Conversational interruption.** Let the user speak naturally over a reply, stop playback promptly, and capture the interruption as the next turn. Account for speaker echo so the assistant’s own voice does not trigger an interruption.
7. **Desktop service.** Start at graphical login through a systemd user service. Persistent sessions are planned separately from the current in-memory history.

## Start at graphical login

After configuring and testing `./voice-assistant`, install the user unit. Its launcher path assumes the checkout is at `~/Dev/projects/python/voice-assistant`. Adjust `ExecStart` in the installed unit if your checkout is elsewhere.

```bash
install -Dm644 systemd/voice-assistant.service ~/.config/systemd/user/voice-assistant.service
systemctl --user daemon-reload
systemctl --user enable --now voice-assistant.service
```

Close any manually started assistant before using the service. Use `systemctl --user stop voice-assistant.service` to stop it and `journalctl --user -u voice-assistant.service -f` to see its output, including transcripts and replies.

## Agent access

Codex defaults to read-only access. To enable full access without approval prompts, set `codex_full_access = true` under `[agent]` in your local `config.toml` and restart the app. Codex can then read and modify files and run commands anywhere your Linux account has permission. The working directory is where it starts. Claude Code runs with its tools disabled.

## Layout and tests

- `src/voice_assistant/agents/` runs the agent CLIs and extracts their final replies.
- `src/voice_assistant/core/` loads TOML settings and coordinates conversation history.
- `src/voice_assistant/audio/` records from PipeWire.
- `src/voice_assistant/stt/` transcribes through whisper.cpp.
- `src/voice_assistant/tts/` synthesizes replies with Kokoro and plays them through PipeWire.
- `tests/` contains offline checks.

```bash
python -m unittest discover -s tests -v
```
