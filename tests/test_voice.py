from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from voice_assistant.main import main
from voice_assistant.stt.whisper_cpp import TranscriptionError


class VoiceTests(unittest.TestCase):
    def test_local_commands_match_complete_phrases_and_only_unmatched_turns_use_agent(self):
        with TemporaryDirectory() as directory:
            config = Path(directory) / "config.toml"
            config.with_name("model.bin").touch()
            config.write_text(
                '[agent]\nprovider = "claude"\n'
                '[stt]\nmodel = "model.bin"\n'
                '[[commands]]\nactivation = ["shutdown now", "shut down now"]\n'
                'code = "bash: echo ran >> tested"\nreply = "Running local command."\n'
                '[[commands]]\nactivation = "test failure"\n'
                'code = "python: raise SystemExit(3)"\n',
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
                    "  Shut   Down   NOW! ", "shutdown now", "What does shutdown now do?", "test failure",
                ]),
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
            self.assertTrue(ask.call_args.args[0].endswith("What does shutdown now do?"))
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
