import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


PROJECT = Path(__file__).resolve().parents[1]


class PortableTests(unittest.TestCase):
    def test_codex_login_and_ask_use_bundle_state_after_move(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "first location"
            bundle.mkdir()
            launcher = bundle / "voice-assistant"
            shutil.copyfile(PROJECT / "portable" / "voice-assistant", launcher)
            launcher.chmod(0o755)
            shutil.copyfile(PROJECT / "portable" / "launch.py", bundle / "launch.py")
            shutil.copytree(PROJECT / "src", bundle / "src")
            python = bundle / "runtime" / "python" / "bin" / "python3"
            python.parent.mkdir(parents=True)
            python.symlink_to(Path(sys.executable).resolve())
            codex = bundle / "runtime" / "codex" / "bin" / "codex"
            codex.parent.mkdir(parents=True)
            codex.write_text(
                f"#!{sys.executable}\n"
                "import json, os, sys\n"
                "from pathlib import Path\n"
                "state = Path(os.environ['CODEX_HOME'])\n"
                "assert 'OPENAI_API_KEY' not in os.environ\n"
                "if sys.argv[1:] == ['login']:\n"
                "    (state / 'auth.json').write_text('fake credentials')\n"
                "else:\n"
                "    assert (state / 'auth.json').read_text() == 'fake credentials'\n"
                "    assert Path.cwd().name == 'workspace'\n"
                "    assert Path(os.environ['TMPDIR']).parent == state.parent\n"
                "    print(json.dumps({'type': 'item.completed', 'item': "
                "{'type': 'agent_message', 'text': sys.stdin.read()}}))\n",
                encoding="utf-8",
            )
            codex.chmod(0o755)
            (bundle / "workspace").mkdir()
            (bundle / "config.toml").write_text(
                '[agent]\nworking_directory = "workspace"\n', encoding="utf-8",
            )
            env = dict(os.environ, CODEX_HOME=str(root / "host state"), OPENAI_API_KEY="fake host key")
            subprocess.run([launcher, "login"], cwd=root, env=env, check=True, timeout=10)
            self.assertTrue((bundle / ".state" / "codex" / "auth.json").is_file())

            moved = bundle.rename(root / "second location")
            reply = subprocess.run(
                [moved / launcher.name, "ask", "Portable reply"], cwd=root, env=env,
                check=True, text=True, capture_output=True, timeout=10,
            )
            self.assertEqual(reply.stdout.strip(), "Portable reply")
            self.assertFalse((root / "host state").exists())
            self.assertFalse((moved / ".state" / "bin" / "claude").exists())
