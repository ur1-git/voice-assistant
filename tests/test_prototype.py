import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from voice_assistant.agents.claude import ClaudeAgent
from voice_assistant.agents.codex import CodexAgent
from voice_assistant.agents.base import AgentError
from voice_assistant.core.config import load_config
from voice_assistant.core.pipeline import Pipeline


class FakeAgent:
    def __init__(self):
        self.prompts = []

    def ask(self, prompt):
        self.prompts.append(prompt)
        return "reply " + str(len(self.prompts))


class PrototypeTests(unittest.TestCase):
    def test_pipeline_history(self):
        fake = FakeAgent()
        pipeline = Pipeline(fake, 2)
        self.assertEqual(pipeline.respond("first"), "reply 1")
        pipeline.respond("second")
        self.assertIn("first", fake.prompts[1])
        self.assertIn("Current user message:\nsecond", fake.prompts[1])

    def test_pipeline_blank_rejected(self):
        with self.assertRaises(ValueError):
            Pipeline(FakeAgent()).respond("   ")

    def test_config_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = load_config(Path(directory) / "missing.toml")
            self.assertEqual(cfg.provider, "codex")
            self.assertEqual(cfg.working_directory, Path(directory))

    def test_codex_events(self):
        payload = "\n".join([
            json.dumps({"type": "thread.started", "thread_id": "x"}),
            json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "Hello!"}}),
        ])
        agent = CodexAgent("codex", Path.cwd())
        with patch.object(agent, "run_process", return_value=payload) as run:
            self.assertEqual(agent.ask("hi"), "Hello!")
            self.assertIn("read-only", run.call_args.args[0])

    def test_claude_json(self):
        agent = ClaudeAgent("claude", Path.cwd())
        with patch.object(agent, "run_process", return_value=json.dumps({"result": "Hello!", "is_error": False})) as run:
            self.assertEqual(agent.ask("hi"), "Hello!")
            self.assertIn("--tools", run.call_args.args[0])

    def test_codex_missing_message(self):
        agent = CodexAgent("codex", Path.cwd())
        with patch.object(agent, "run_process", return_value='{"type":"turn.completed"}'):
            with self.assertRaises(AgentError):
                agent.ask("hi")


if __name__ == "__main__":
    unittest.main()
