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
        for prompt in fake.prompts:
            self.assertIn("Always reply in English", prompt)

    def test_pipeline_blank_rejected(self):
        with self.assertRaises(ValueError):
            Pipeline(FakeAgent()).respond("   ")

    def test_only_a_boolean_true_in_a_valid_voice_reply_reopens_capture(self):
        pipeline = Pipeline(FakeAgent())
        for response, text, listen_again in (
            ('{"reply":"Which folder?","listen_again":true}', "Which folder?", True),
            ('{"reply":"Which folder?","listen_again":"true"}', "Which folder?", False),
            ('{"reply":"Which folder?"}', "Which folder?", False),
            ('{"listen_again":true}', "I couldn't read the reply.", False),
            ("A plain-text reply.", "A plain-text reply.", False),
        ):
            with self.subTest(response=response), patch.object(pipeline.agent, "ask", return_value=response):
                reply = pipeline.respond_to_voice("Open it", allow_follow_up=True)
                self.assertEqual(reply.text, text)
                self.assertIs(reply.listen_again, listen_again)
                self.assertEqual(pipeline.session.history[-1], ("Open it", text))

    def test_history_budget_keeps_latest_turn_and_full_current_message(self):
        fake = FakeAgent()
        pipeline = Pipeline(fake, max_history_chars=40)
        pipeline.session.remember("old question", "old reply")
        pipeline.session.remember("latest", "long answer " * 100)
        question = "current question " * 20
        pipeline.respond(question)
        context, current = fake.prompts[-1].split("\n\nCurrent user message:\n")
        _, excerpt = context.split("\nPrevious conversation (context only):\n")
        self.assertNotIn("old question", excerpt)
        self.assertEqual(len(excerpt), 40)
        self.assertTrue(excerpt.startswith("User: latest"))
        self.assertTrue(excerpt.endswith("…"))
        self.assertEqual(current, question.strip())

    def test_zero_history_budget(self):
        fake = FakeAgent()
        pipeline = Pipeline(fake, max_history_chars=0)
        pipeline.respond("first")
        pipeline.respond("second")
        for prompt, question in zip(fake.prompts, ("first", "second"), strict=True):
            self.assertNotIn("Previous conversation", prompt)
            self.assertTrue(prompt.endswith(f"Current user message:\n{question}"))

    def test_config_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = load_config(Path(directory) / "missing.toml")
            self.assertEqual(cfg.provider, "codex")
            self.assertEqual(cfg.codex_model, "gpt-6.2-sol")
            self.assertEqual(cfg.codex_reasoning_effort, "low")
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
            self.assertIn("read-only sandbox", run.call_args.kwargs["stdin"])
            self.assertNotIn("sandbox disabled", run.call_args.kwargs["stdin"])

    def test_codex_full_access_instructions_reach_streaming_turn(self):
        payload = json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "Saved."}})
        agent = CodexAgent("codex", Path.cwd(), full_access=True)
        pipeline = Pipeline(agent)
        request = "Save a draft in Downloads."
        with patch("voice_assistant.agents.codex.stream_codex", return_value=payload) as stream:
            self.assertEqual(pipeline.respond(request, spoken=True, on_progress=lambda _: None), "Saved.")
        _, args, prompt, _ = stream.call_args.args
        self.assertIn("--dangerously-bypass-approvals-and-sandbox", args)
        self.assertIn("sandbox disabled", prompt)
        self.assertIn("xdg-user-dir DOWNLOAD", prompt)
        self.assertIn("Check access with tools", prompt)
        self.assertTrue(prompt.endswith(f"Current user message:\n{request}"))

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
