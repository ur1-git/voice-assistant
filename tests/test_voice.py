from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Event
from time import sleep
import unittest
from unittest.mock import MagicMock, patch
import wave

from voice_assistant.audio.playback import is_stop_request, listen_for_interruption
from voice_assistant.core.commands import load_commands
from voice_assistant.core.config import load_voice_config
from voice_assistant.main import main
from voice_assistant.stt.whisper_cpp import TranscriptionError


class VoiceTests(unittest.TestCase):
    def setUp(self):
        capture = patch("voice_assistant.audio.playback.pcm_frames", side_effect=lambda *a, **kw: (_ for _ in ()))
        capture.start()
        self.addCleanup(capture.stop)

    def test_stop_matches_only_the_final_words(self):
        for transcript in ("stop now", "Please STOP   NOW!", "Hey Nova, stop, now…"):
            with self.subTest(transcript=transcript):
                self.assertTrue(is_stop_request(transcript))
        for transcript in ("stop now and open Firefox", "nonstop now", "stop nowadays"):
            with self.subTest(transcript=transcript):
                self.assertFalse(is_stop_request(transcript))

    def test_recorded_stop_cancels_configured_command_before_dispatch(self):
        with TemporaryDirectory() as directory:
            config = Path(directory) / "config.toml"
            config.with_name("model.bin").touch()
            config.write_text(
                '[stt]\nmodel = "model.bin"\n'
                '[[commands]]\nactivation = "open firefox and stop now"\n'
                'codeBash = "touch tested"\n',
                encoding="utf-8",
            )
            stdout = StringIO()
            with (
                patch("shutil.which", return_value="/fake/tool"),
                patch("builtins.input", side_effect=["", "/quit"]),
                patch("voice_assistant.main.record_wav"),
                patch("voice_assistant.main.transcribe", return_value="Open Firefox and stop now!"),
                patch("voice_assistant.main.LearnedCommands") as learned,
                patch("voice_assistant.main.CodexAgent.ask") as ask,
                patch("voice_assistant.main.KokoroSpeaker") as speaker,
                redirect_stdout(stdout),
            ):
                self.assertEqual(main(["--config", str(config), "voice", "--speak"]), 0)

            self.assertFalse(config.with_name("tested").exists())
            learned.return_value.lookup.assert_not_called()
            ask.assert_not_called()
            speaker.return_value.speak.assert_called_once_with("Stopped.", interrupt=None)
            self.assertIn("Assistant> Stopped.", stdout.getvalue())

    def test_playback_stop_cancels_progress_turn_without_adding_history(self):
        with TemporaryDirectory() as directory:
            config = Path(directory) / "config.toml"
            config.with_name("model.bin").touch()
            config.write_text('[stt]\nmodel = "model.bin"\n', encoding="utf-8")
            stdout = StringIO()
            playing = Event()

            def ask(prompt, *, on_progress, check_interruption):
                if prompt.endswith("Cancel this request"):
                    on_progress("I heard you.")
                    self.fail("The stop phrase should cancel the running turn")
                self.assertNotIn("Cancel this request", prompt)
                return "Second request completed."

            def speak(message, *, interrupt):
                if message == "I heard you.":
                    playing.set()
                    return not interrupt(lambda: True)
                return True

            def listen(*args, listening, playing_text):
                while listening():
                    if playing.wait(0.01) and playing_text() == "I heard you.":
                        return "stop"

            with (
                patch("shutil.which", return_value="/fake/tool"),
                patch("builtins.input", side_effect=["", "", "/quit"]),
                patch("voice_assistant.main.record_wav"),
                patch("voice_assistant.main.transcribe", side_effect=["Cancel this request", "Second request"]),
                patch("voice_assistant.audio.playback.listen_for_interruption", side_effect=listen),
                patch("voice_assistant.main.CodexAgent.ask", side_effect=ask),
                patch("voice_assistant.main.KokoroSpeaker") as speaker,
                redirect_stdout(stdout),
            ):
                speaker.return_value.speak.side_effect = speak
                self.assertEqual(main(["--config", str(config), "voice", "--speak"]), 0)

            self.assertEqual(
                [call.args[0] for call in speaker.return_value.speak.call_args_list],
                ["I heard you.", "Stopped.", "Second request completed."],
            )

    def test_playback_recognizes_the_stop_suffix_after_the_wake_phrase(self):
        config = load_voice_config(Path("config.toml"), wake=True)
        wake = MagicMock()
        silence = bytes(2560)
        voice = (1000).to_bytes(2, "little", signed=True) * 1280

        def frames(*args, **kwargs):
            yield from [silence] * 2 + [voice] * 8 + [silence] * 4

        def transcribe_clip(path, **kwargs):
            with wave.open(str(path), "rb") as clip:
                self.assertEqual((clip.getframerate(), clip.getnchannels(), clip.getsampwidth()), (16000, 1, 2))
                self.assertEqual(clip.getnframes(), 14 * 1280)
            return "Hey Nova, stop now."

        wake.detected.side_effect = [False] * 4 + [True]
        with (
            patch("voice_assistant.audio.playback.pcm_frames", side_effect=frames),
            patch("voice_assistant.audio.playback.transcribe", side_effect=transcribe_clip),
        ):
            self.assertEqual(
                listen_for_interruption(config, wake, None, listening=lambda: True, playing_text=lambda: "A long reply"),
                "stop",
            )

    def test_stop_interrupts_a_silent_agent_process(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.toml"
            config.write_text('[stt]\nmodel = "model.bin"\n', encoding="utf-8")
            (root / "model.bin").touch()
            marker = root / "running"
            codex = root / "codex"
            codex.write_text(
                f'#!{sys.executable}\n'
                'import os, time\nfrom pathlib import Path\n'
                'Path("running").write_text(str(os.getpid()))\n'
                'time.sleep(3)\nPath("finished").touch()\n',
                encoding="utf-8",
            )
            codex.chmod(0o700)
            stdout = StringIO()

            def listen(*args, listening, playing_text):
                while listening():
                    if marker.is_file() and marker.read_text():
                        self.assertEqual(playing_text(), "")
                        return "stop"
                    sleep(0.01)

            with (
                patch("shutil.which", return_value="/fake/tool"),
                patch("builtins.input", side_effect=["", "/quit"]),
                patch("voice_assistant.main.record_wav"),
                patch("voice_assistant.main.transcribe", return_value="Start a request"),
                patch("voice_assistant.audio.playback.listen_for_interruption", side_effect=listen),
                patch("voice_assistant.main.KokoroSpeaker"),
                redirect_stdout(stdout),
            ):
                self.assertEqual(main(
                    ["--config", str(config), "voice", "--speak"], agent_executables={"codex": str(codex)},
                ), 0)

            self.assertIn("Assistant> Stopped.", stdout.getvalue())
            self.assertFalse((root / "finished").exists())

    def test_one_wake_during_playback_opens_a_new_recording(self):
        with TemporaryDirectory() as directory:
            config = Path(directory) / "config.toml"
            config.write_text('[stt]\nmodel = "model.bin"\n', encoding="utf-8")
            config.with_name("model.bin").touch()
            playing = Event()

            def listen(*args, listening, playing_text):
                while listening():
                    if playing.wait(0.01) and playing_text():
                        return "wake"

            def speak(message, *, interrupt):
                playing.set()
                return not interrupt(lambda: True)

            with (
                patch("shutil.which", return_value="/fake/tool"),
                patch("voice_assistant.main.WakeWordDetector") as wake,
                patch("voice_assistant.main.EndOfSpeechRecorder") as recorder,
                patch("voice_assistant.main.play_chime") as chime,
                patch("voice_assistant.main.transcribe", return_value="Start a request"),
                patch("voice_assistant.main.CodexAgent.ask", return_value="A long reply."),
                patch("voice_assistant.main.KokoroSpeaker") as speaker,
                patch("voice_assistant.audio.playback.listen_for_interruption", side_effect=listen),
                redirect_stdout(StringIO()),
            ):
                wake.return_value.wait.side_effect = [True, KeyboardInterrupt]
                recorder.return_value.record.side_effect = [True, False]
                speaker.return_value.speak.side_effect = speak
                self.assertEqual(main(["--config", str(config), "voice", "--wake", "--speak"]), 0)

            self.assertEqual(wake.return_value.wait.call_count, 2)
            self.assertEqual([call.kwargs["waking"] for call in chime.call_args_list], [True, False, True, False])
            self.assertEqual(
                [call.kwargs["start_timeout_seconds"] for call in recorder.return_value.record.call_args_list],
                [5, 5],
            )

    def test_command_fields_accept_legacy_syntax_and_reject_ambiguity(self):
        for field, language, code in (
            ("codeBash", "bash", "echo 'Hello: world'"),
            ("codePython", "python", "print('Hello: world')"),
        ):
            with self.subTest(field=field):
                legacy = load_commands([{"activation": "test", "code": f"{language}: {code}"}])
                current = load_commands([{"activation": "test", field: code}])
                self.assertEqual(current, legacy)
        for fields in ({}, {"codeBash": "true", "codePython": "pass"}, {"code": "bash: true", "codeBash": "true"}):
            with self.subTest(fields=fields), self.assertRaisesRegex(ValueError, "exactly one"):
                load_commands([{"activation": "test", **fields}])

    def test_run_before_reply_requires_a_boolean(self):
        with self.assertRaisesRegex(ValueError, "runBeforeReply.*true or false"):
            load_commands([{"activation": "test", "codeBash": "true", "runBeforeReply": "false"}])

    def test_commands_announce_the_result_after_execution_by_default(self):
        with TemporaryDirectory() as directory:
            config = Path(directory) / "config.toml"
            marker = config.with_name("tested")
            config.with_name("model.bin").touch()
            config.write_text(
                '[stt]\nmodel = "model.bin"\n'
                '[[commands]]\nactivation = "test success"\n'
                'codeBash = "touch tested"\nreply = "It worked."\n'
                '[[commands]]\nactivation = "test failure"\n'
                'codePython = "raise SystemExit(3)"\nrunBeforeReply = true\nreply = "False success."\n',
                encoding="utf-8",
            )
            replies = []

            def speak(message, *, interrupt):
                self.assertTrue(marker.exists())
                replies.append(message)
                return True

            with (
                patch("shutil.which", return_value="/fake/tool"),
                patch("builtins.input", side_effect=["", "", "/quit"]),
                patch("voice_assistant.main.record_wav"),
                patch("voice_assistant.main.transcribe", side_effect=["test success", "test failure"]),
                patch("voice_assistant.main.KokoroSpeaker") as speaker,
                redirect_stdout(StringIO()),
                redirect_stderr(StringIO()),
            ):
                speaker.return_value.speak.side_effect = speak
                self.assertEqual(main(["--config", str(config), "voice", "--speak"]), 0)

            self.assertEqual(replies, ["It worked.", "The local command failed."])

    def test_app_launch_is_learned_and_replays_without_ai_after_restart(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.toml"
            config.write_text('[stt]\nmodel = "model.bin"\n', encoding="utf-8")
            config.with_name("model.bin").touch()
            applications = root / "applications"
            applications.mkdir()
            desktop = applications / "test-music.desktop"
            desktop.write_text(
                "[Desktop Entry]\nType=Application\nName=Test Music (Launcher)\nExec=/usr/bin/true\n",
                encoding="utf-8",
            )
            stdout = StringIO()
            with (
                patch.dict(os.environ, {"XDG_DATA_HOME": directory, "XDG_STATE_HOME": directory}),
                patch("shutil.which", side_effect=lambda tool: f"/fake/{tool}"),
                patch("builtins.input", side_effect=["", "", "", "/quit", "", "/quit"]),
                patch("voice_assistant.main.record_wav"),
                patch("voice_assistant.main.transcribe", side_effect=[
                    "Open Test Music", "Open Test Music.", "OPEN Test Music!", "Open Test Music.",
                ]),
                patch("voice_assistant.main.CodexAgent.ask", side_effect=[
                    "LAUNCH_DESKTOP_APP invented.desktop", "LAUNCH_DESKTOP_APP test-music.desktop",
                ]) as ask,
                patch("voice_assistant.core.commands.subprocess.run", return_value=subprocess.CompletedProcess([], 0)) as run,
                redirect_stdout(stdout),
            ):
                arguments = ["--config", str(config), "voice"]
                self.assertEqual(main(arguments), 0)
                self.assertEqual(main(arguments), 0)

            self.assertEqual(ask.call_count, 2)
            self.assertIn('"test-music.desktop": "Test Music (Launcher)"', ask.call_args.args[0])
            self.assertEqual(run.call_count, 3)
            for launch in run.call_args_list:
                self.assertEqual(launch.args[0], ("/fake/gio", "launch", str(desktop)))
                self.assertEqual(launch.kwargs["stdout"], subprocess.DEVNULL)
                self.assertEqual(launch.kwargs["stderr"], subprocess.DEVNULL)
            self.assertIn("I couldn't match that request to an installed app.", stdout.getvalue())

    def test_local_commands_match_complete_phrases_and_only_unmatched_turns_use_agent(self):
        with TemporaryDirectory() as directory:
            config = Path(directory) / "config.toml"
            config.with_name("model.bin").touch()
            config.write_text(
                '[agent]\nprovider = "claude"\n'
                '[stt]\nmodel = "model.bin"\n'
                '[[commands]]\nactivation = ["shutdown now", "shut down now"]\n'
                'codeBash = "echo ran >> tested"\nrunBeforeReply = false\nreply = "Running local command."\n'
                '[[commands]]\nactivation = "test failure"\n'
                'codePython = "raise SystemExit(3)"\n',
                encoding="utf-8",
            )
            stdout, stderr = StringIO(), StringIO()
            announcements = []

            def speak(message, *, interrupt):
                if message != "Running local command.":
                    return True
                announcements.append(message)
                if len(announcements) == 1:
                    self.assertFalse(config.with_name("tested").exists())
                return len(announcements) == 1

            with (
                patch("shutil.which", return_value="/fake/tool"),
                patch("builtins.input", side_effect=["", "", "", "", "/quit"]),
                patch("voice_assistant.main.record_wav"),
                patch("voice_assistant.main.transcribe", side_effect=[
                    "  Shut   Down   NOW! ", "shutdown now", "Você pode explicar?", "test failure",
                ]) as transcribe,
                patch("voice_assistant.main.KokoroSpeaker") as speaker,
                patch("voice_assistant.main.ClaudeAgent.ask", return_value="An explanation.") as ask,
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                speaker.return_value.speak.side_effect = speak
                self.assertEqual(main(["--config", str(config), "voice", "--speak"]), 0)

            self.assertEqual(config.with_name("tested").read_text(), "ran\n")
            self.assertEqual(len(announcements), 2)
            self.assertEqual(ask.call_count, 1)
            prompt = ask.call_args.args[0]
            self.assertIn("Always reply in English", prompt)
            self.assertIn("Your entire reply will be read aloud", prompt)
            self.assertTrue(prompt.endswith("Current user message:\nVocê pode explicar?"))
            self.assertEqual(transcribe.call_args.kwargs["language"], "en")
            self.assertIn("Local command canceled.", stdout.getvalue())
            self.assertIn("The local command failed.", stdout.getvalue())
            self.assertIn("Local command exited 3.", stderr.getvalue())

    def test_voice_chat_keeps_history_and_cleans_up_failed_turns(self):
        with TemporaryDirectory() as directory:
            config = Path(directory) / "config.toml"
            model = config.with_name("model.bin")
            model.touch()
            config.write_text(
                '[agent]\nprovider = "claude"\n'
                '[audio]\ndevice = "my-microphone"\nmax_record_seconds = 15\n'
                '[stt]\nmodel = "model.bin"\nlanguage = "es"\ntimeout_seconds = 45\n',
                encoding="utf-8",
            )
            stdout = StringIO()
            stderr = StringIO()
            with (
                patch("shutil.which", return_value="/fake/tool"),
                patch("builtins.input", side_effect=["", "", "", "", "/quit"]),
                patch("voice_assistant.main.record_wav") as record,
                patch("voice_assistant.main.transcribe", side_effect=[
                    "First request", TranscriptionError("timeout"), "", "Follow-up",
                ]) as transcribe,
                patch("voice_assistant.main.ClaudeAgent.ask", return_value="Agent reply") as ask,
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                self.assertEqual(main(["--config", str(config), "voice"]), 0)

            self.assertIn("Transcript> First request", stdout.getvalue())
            self.assertIn("Assistant> Agent reply", stdout.getvalue())
            self.assertIn("No speech transcribed.", stdout.getvalue())
            self.assertIn("Error: timeout", stderr.getvalue())
            self.assertEqual(ask.call_count, 2)
            follow_up = ask.call_args.args[0]
            self.assertIn("User: First request", follow_up)
            self.assertIn("Current user message:\nFollow-up", follow_up)

            for capture, transcription in zip(record.call_args_list, transcribe.call_args_list):
                audio = capture.args[0]
                self.assertEqual(transcription.args[0], audio)
                self.assertEqual(capture.kwargs, {"device": "my-microphone", "max_seconds": 15})
                self.assertEqual(transcription.kwargs, {
                    "model": model, "language": "es", "timeout_seconds": 45,
                })
                self.assertFalse(audio.parent.exists())

    def test_wake_follow_up_reuses_recorder_and_returns_to_idle(self):
        with TemporaryDirectory() as directory:
            config = Path(directory) / "config.toml"
            config.with_name("model.bin").touch()
            config.write_text('[agent]\nprovider = "claude"\n[stt]\nmodel = "model.bin"\n', encoding="utf-8")
            stdout = StringIO()
            with (
                patch("shutil.which", return_value="/fake/tool"),
                patch("voice_assistant.main.WakeWordDetector") as wake,
                patch("voice_assistant.main.EndOfSpeechRecorder") as recorder,
                patch("voice_assistant.main.KokoroSpeaker") as speaker,
                patch("voice_assistant.main.play_chime"),
                patch("voice_assistant.main.transcribe", side_effect=[
                    "First request", "Downloads", "Second request", "Third request",
                ]) as transcribe,
                patch("voice_assistant.main.ClaudeAgent.ask", side_effect=[
                    '{"reply":"Which folder?","listen_again":true}',
                    '{"reply":"Done.","listen_again":false}',
                    '{"reply":"What format?","listen_again":true}',
                    '{"reply":"Which file?","listen_again":true}',
                ]) as ask,
                redirect_stdout(stdout),
            ):
                wake.return_value.wait.side_effect = [True, True, True, KeyboardInterrupt]
                speaker.return_value.speak.side_effect = [True, True, True, False]
                recordings = iter([True, True, True, False, True])
                wake_counts = []

                def record(*args, **kwargs):
                    wake_counts.append(wake.return_value.wait.call_count)
                    return next(recordings)

                recorder.return_value.record.side_effect = record
                self.assertEqual(main(["--config", str(config), "voice", "--wake", "--speak"]), 0)

            self.assertEqual(wake_counts, [1, 1, 2, 2, 3])
            self.assertEqual(
                [call.kwargs["start_timeout_seconds"] for call in recorder.return_value.record.call_args_list],
                [5, 3, 5, 3, 5],
            )
            self.assertEqual(transcribe.call_count, 4)
            self.assertIn("Assistant: Which folder?", ask.call_args_list[1].args[0])
            self.assertIn("Current user message:\nDownloads", ask.call_args_list[1].args[0])
            self.assertEqual(
                [call.args[0] for call in speaker.return_value.speak.call_args_list],
                ["Which folder?", "Done.", "What format?", "Which file?"],
            )
            self.assertIn("No speech detected.", stdout.getvalue())
