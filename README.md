# Linux voice assistant

Say “Hey Jarvis,” wait for the chime, then speak your request. Whisper transcribes it on your machine, Codex answers, and Kokoro reads the reply aloud. You can also type requests or start microphone recording with Enter. Claude Code is an optional alternative to Codex.

Speech recognition, wake detection, and speech synthesis run locally. The selected agent needs an internet connection and receives the transcript and recent conversation. Codex can also read files in the configured project.

- [Install from source](#install-from-source) for the full hands-free setup.
- [Use the portable archive](#use-the-portable-archive) for text and push-to-talk with fewer host dependencies.
- [Train a custom wake phrase](#train-a-custom-wake-phrase) to use a different phrase or add your own recordings.

## Install from source

Use a Linux desktop with a running PipeWire session and Python 3.11 or newer. If your tools are already installed, you can reuse them. For a fresh Arch Linux setup with Codex and the system Whisper executable, install the required packages with this command. [PipeWire audio tools](https://archlinux.org/packages/extra/x86_64/pipewire-audio/files/) · [whisper.cpp](https://archlinux.org/packages/extra/x86_64/whisper-cpp/files/) · [Codex](https://archlinux.org/packages/extra/x86_64/openai-codex/)

```bash
sudo pacman -S --needed git python curl pipewire-audio wireplumber whisper-cpp openai-codex
```

On other distributions, install Git, Python with venv support, curl, and your distribution's PipeWire tools (`pw-record` and `pw-play`). Install Codex using its upstream instructions and either use a packaged `whisper-cli` or [build it inside the project](#build-whispercpp-inside-the-project). [Codex CLI](https://github.com/openai/codex#readme)

### 1. Set up the app

In a terminal, clone the repository into the directory where you want to keep it. If you already have a checkout, open a terminal there and start at `python3 -m venv .venv`.

```bash
git clone https://github.com/ur1-git/voice-assistant.git
cd voice-assistant
python3 -m venv .venv
.venv/bin/python -m pip install --no-cache-dir -e .
test -e config.toml || cp config.example.toml config.toml
./voice-assistant doctor
```

`doctor` lists commands it can find. For example, Claude can be missing if you're using Codex. Sign in if needed and try a typed request.

```bash
codex login
./voice-assistant ask "What does this project do?"
```

Set `agent.working_directory` in `config.toml` to the directory you want the agent to work in. The default `"."` points to the assistant's folder. Relative paths in the config start at the directory containing the TOML file. To choose another config, run `./voice-assistant --config /path/to/settings.toml chat`.

If you use Claude Code, install and sign in to its CLI, then set `agent.provider = "claude"` or pass `--agent claude` to a command. [Claude Code setup](https://code.claude.com/docs/en/setup)

### 2. Enable microphone input

The app uses the `whisper-cli` executable and a separate GGML model. If you already have a model, set `stt.model` in `config.toml` to its path. Otherwise run the first command to download the default multilingual Whisper Small model (about 488 MB) into `models/`. The downloader keeps existing files at their destination paths. [Whisper Small weights](https://huggingface.co/ggerganov/whisper.cpp/blob/main/ggml-small.bin)

```bash
./scripts/download-models whisper
./voice-assistant voice
```

Press Enter to start recording. Speak, then press Ctrl+C to send the recording. Type `/quit` at the prompt to exit. The app captures up to 30 seconds by default. Input defaults to English. Set `stt.language = "auto"` with a multilingual model for language detection. Agent replies are requested in English.

### 3. Enable spoken replies

Install the Kokoro Python dependencies and download the speech model and voice data (about 354 MB combined). [Kokoro model release](https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.1)

```bash
.venv/bin/python -m pip install --no-cache-dir -e '.[tts]'
./scripts/download-models kokoro
./voice-assistant voice --speak
```

Kokoro speaks with the `af_heart` voice by default. Change `tts.voice` or `tts.language` in `config.toml` to select a voice and pronunciation language. With Codex, the assistant can also speak short progress updates from the current request.

### 4. Enable hands-free mode

Install the ONNX dependencies separately because openWakeWord 0.6.0 also declares a TensorFlow Lite dependency on Linux. Download the Hey Jarvis classifier, its two feature models, and Silero VAD. [openWakeWord installation](https://github.com/dscripka/openWakeWord/tree/v0.6.0#installation) · [ONNX model release](https://github.com/dscripka/openWakeWord/releases/tag/v0.5.1)

```bash
.venv/bin/python -m pip install --no-cache-dir numpy 'onnxruntime>=1.10,<2' 'requests>=2,<3' 'scipy>=1.3,<2' 'scikit-learn>=1,<2' 'tqdm>=4,<5'
.venv/bin/python -m pip install --no-cache-dir --no-deps 'openwakeword==0.6.0'
./scripts/download-models wake
./voice-assistant
```

To fetch all the default speech models together, run `./scripts/download-models all`. Each set can also be downloaded separately as shown above.

Say “Hey Jarvis,” wait for the rising chime, and start speaking within five seconds. A falling chime signals that recording ended. The default pause before sending is one second. If the assistant needs a follow-up, wait for the rising chime and start speaking again within three seconds. Ctrl+C exits while waiting for the wake phrase.

### Stop a voice request or reply

Say **“stop now”** while the assistant is speaking or working silently, or end your recorded request with those words. The assistant says **“Stopped.”** A recorded request ending in “stop now” is canceled before any command or AI request. This works with “please stop now” too. A phrase like “stop now and open Firefox” follows the normal request flow.

Say the wake phrase while the assistant is busy to interrupt and open a fresh recording. Wait for the rising chime before speaking your new request. Ctrl+C also interrupts playback. Headphones help keep playback out of the microphone. If the reply itself says “stop now,” use the wake phrase or say “please stop now” so the assistant can distinguish the interruption from its own words.

### Build whisper.cpp inside the project

If you prefer a project-local executable, use CMake and a C/C++ toolchain to build it. Run these commands from the checkout. The launcher searches `bin/` before your usual `PATH`. [whisper.cpp build instructions](https://github.com/ggml-org/whisper.cpp#quick-start)

```bash
git clone --depth 1 --branch v1.9.5 https://github.com/ggml-org/whisper.cpp.git .build/whisper.cpp
cmake -S .build/whisper.cpp -B .build/whisper-build \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
    -DWHISPER_BUILD_TESTS=OFF -DWHISPER_BUILD_SERVER=OFF
cmake --build .build/whisper-build --target whisper-cli --parallel
install -Dm755 .build/whisper-build/bin/whisper-cli bin/whisper-cli
./voice-assistant doctor
```

### Troubleshooting

| If you see this | Check this |
| --- | --- |
| A missing executable in `doctor` | Check `whisper-cli` and `pw-record` for microphone input. Spoken replies and wake chimes also need `pw-play`. |
| The wrong microphone is recording | Run `wpctl status --name`. Set `audio.device` to the node name under Sources. |
| A model file is missing | Check the path in `config.toml` and run the corresponding model download command. If a file is corrupt, move it aside and rerun the downloader. |
| The wake phrase fires too often or rarely | Raise `wakeword.threshold` above `0.5` for false wakes. Lower it in small steps for missed wakes and retest. |
| The assistant sends before you're done | Increase `audio.end_silence_ms` and restart. |

## Train a custom wake phrase

If you already have an openWakeWord ONNX model, skip to [load the model](#load-the-model).

Pick a short, distinct English phrase such as “Hey Nova.” For a basic model generated from text, enter your phrase in the upstream Colab notebook and run the training cells. Download the resulting `.onnx` file. [Quick training notebook](https://colab.research.google.com/drive/1q1oe2zOyZp7UsB3jJiQ1IFn8z5YfjwEb?usp=sharing)

### Include your own recordings

1. Open the detailed upstream notebook in a separate Linux training environment. You can keep the notebook, data, and outputs in `wakeword-training/` inside this checkout. Run the notebook's setup, data download, and configuration cells. Set `target_phrase` to your phrase and run the synthetic clip generation cell. The synthetic clips and upstream negative datasets give the model more variety than a few recordings from one speaker. [Detailed training notebook](https://github.com/dscripka/openWakeWord/blob/v0.6.0/notebooks/automatic_model_training.ipynb)
2. Before running the augmentation cell, copy your WAVs into the generated clip folders under `output_dir/model_name/` (for example, `my_custom_model/hey_nova/`). Positive clips contain one complete wake phrase. Negative clips can include similar phrases, ordinary speech, and sounds that should leave the assistant idle. Keep each recording session in one split to avoid testing on the same takes used in training. [Training script](https://github.com/dscripka/openWakeWord/blob/v0.6.0/openwakeword/train.py)

   | Folder | What goes there |
   | --- | --- |
   | `positive_train/` | Your spoken wake phrase at different speeds, distances, and volumes |
   | `negative_train/` | Similar phrases and other short audio without the wake phrase |
   | `positive_test/` | Wake phrase examples from a separate recording session |
   | `negative_test/` | Other speech and sounds from a separate recording session |

   Use uniquely named 16 kHz, mono, signed 16-bit PCM WAV files. Leave a little silence on each side of the spoken phrase. In a terminal, `pw-record --rate=16000 --channels=1 --format=s16 take-01.wav` records one clip until you press Ctrl+C. Record extra clips for a final check after training.
3. Run the augmentation and training cells. If you already ran augmentation for this model, add `--overwrite` to the augmentation command so it rebuilds the cached features with your recordings. The trained file is saved as `output_dir/model_name.onnx`. [ONNX export](https://github.com/dscripka/openWakeWord/blob/v0.6.0/openwakeword/train.py#L875-L889)

### Load the model

Copy the exported file to `models/openwakeword/` alongside `embedding_model.onnx`, `melspectrogram.onnx`, and `silero_vad.onnx`. Edit the existing `[wakeword]` section in `config.toml`.

```toml
[wakeword]
model = "models/openwakeword/hey_nova.onnx"
threshold = 0.5
```

Restart the assistant. It prints the loaded model name and threshold at startup. Test the phrase on your normal microphone, then try similar phrases and ordinary background audio to check for false wakes. To switch back, set the model path to `models/openwakeword/hey_jarvis_v0.1.onnx`.

## Start at graphical login

After testing `./voice-assistant` manually, close it and run these commands from the checkout on a desktop that uses systemd's `graphical-session.target`. Enabling by path links the unit from its `systemd/` folder and lets it find the launcher relative to that location. If you installed an older copy of the unit in your user config, disable it and move that copy aside before linking the new one. [systemd unit paths](https://www.freedesktop.org/software/systemd/man/latest/systemd.unit.html)

```bash
systemctl --user enable --now "$(pwd -P)/systemd/voice-assistant.service"
systemctl --user status voice-assistant.service
journalctl --user -u voice-assistant.service -f
```

The journal includes transcripts and replies. To pause the assistant, run `systemctl --user stop voice-assistant.service`. After editing `config.toml`, restart it with `systemctl --user restart voice-assistant.service`.

## Move or remove a source installation

Keep `.venv/`, `bin/`, `models/`, and `config.toml` inside the checkout. A Python venv contains absolute paths, so recreate it and repeat the pip commands after moving the checkout. Use relative paths in `config.toml` for files you want to move with it. Before moving a checkout with autostart enabled, run `systemctl --user disable --now voice-assistant.service`, then enable the unit again at the new location.

To remove the app, quit it, disable its user service if enabled, and remove the checkout after backing up any config, custom models, or recordings you want to keep. If you used the older autostart instructions, also remove the copy of `voice-assistant.service` from your user systemd config and run `systemctl --user daemon-reload`. Host packages and the agent CLI's normal sign-in and settings can be shared with other tools. Learned shortcuts for a source install are stored per config path in `${XDG_STATE_HOME:-$HOME/.local/state}/voice-assistant/`.

## Use the portable archive

The optional Linux x86-64 archive includes standalone Python, Codex, CPU-based whisper.cpp, and the multilingual Whisper Base model. It supports typed and push-to-talk requests with text replies. The full Kokoro and wake setup is described in [Install from source](#install-from-source). Voice recording uses your desktop PipeWire session and a compatible `pw-record` client.

Extract `voice-assistant-linux-x86_64.tar.gz`, open a terminal in the extracted folder, then run:

```bash
./voice-assistant doctor
./voice-assistant login
./voice-assistant voice
```

The bundle defaults to its own `workspace/` directory. Set `agent.working_directory` in its `config.toml` to change that. For Claude Code, run `./voice-assistant login claude` and `./voice-assistant voice --agent claude`. The bundle keeps agent credentials, caches, and assistant state under `.state/`. To remove it, quit and delete the extracted folder. Share the original archive to keep your config and credentials private.

To build the archive from source on x86-64 Linux, run `python3 portable/build.py` with Python 3.12+, CMake, a C/C++ toolchain, and internet access. It reuses downloads in `dist/.downloads/` and writes the archive under `dist/`. Use `--assets /path/to/downloads` for a separate download cache or `--capture /path/to/recording-prefix` to provide a redistributable `bin/pw-record` with its client libraries and license files. Build on the oldest Linux release you intend to support because the native Whisper executable uses the build host's C library.

## Local voice commands

Add a `[[commands]]` block for each command. The example config already includes volume up and down. This example prints the date to the terminal.

```toml
[[commands]]
activation = ["show the date", "what is the date"]
codeBash = "date"
reply = "I printed the date in the terminal."
```

Use one activation string or a list of alternatives. Set `codeBash` or `codePython` to the code to run. The complete transcript is matched after normalizing case, whitespace, and punctuation at the edges. Commands run as your Linux user from `agent.working_directory`. By default the command runs before its spoken reply. Set `runBeforeReply = false` to speak first and allow an interruption to cancel the pending command.

For a voice shutdown with a ten-minute grace period, put the following entries in `config.toml`. After scheduling, say the wake phrase and then “cancel shutdown” to clear the pending power-off. [systemctl scheduling options](https://www.freedesktop.org/software/systemd/man/latest/systemctl.html#--when=)

```toml
[[commands]]
activation = ["shutdown now", "shut down now"]
codeBash = "/usr/bin/systemctl --no-ask-password --when=+10min poweroff"
runBeforeReply = false
reply = "I’ll schedule shutdown for ten minutes from now."

[[commands]]
activation = "cancel shutdown"
codeBash = "/usr/bin/systemctl --no-ask-password --when=cancel poweroff"
reply = "Shutdown canceled."
```

Say “Open Spotify” or “Launch Firefox” to use an installed desktop launcher. The agent can resolve an unfamiliar phrase to an installed `.desktop` entry. After a successful `gio launch`, the assistant remembers the shortcut and can handle the same phrase locally. Configured commands take precedence.

Codex defaults to read-only project access. Setting `agent.codex_full_access = true` allows Codex to run commands and modify files with your Linux account's permissions without approval prompts. The assistant runs Claude Code with its tools disabled.

## Run the tests

```bash
.venv/bin/python -m unittest discover -s tests -v
```
