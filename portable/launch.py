"""Run an extracted bundle using its own runtime, tools, and state."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
from tempfile import TemporaryDirectory
from urllib.request import urlopen


BUNDLE = Path(__file__).resolve().parent
STATE = BUNDLE / ".state"
CODEX = BUNDLE / "runtime" / "codex" / "bin" / "codex"
CLAUDE = STATE / "bin" / "claude"
CLAUDE_RELEASES = "https://downloads.claude.ai/claude-code-releases"


def setup_environment() -> None:
    os.umask(0o077)
    for key, directory in {
        "CODEX_HOME": "codex",
        "CLAUDE_CONFIG_DIR": "claude",
        "XDG_CACHE_HOME": "cache",
        "XDG_CONFIG_HOME": "config",
        "XDG_DATA_HOME": "data",
        "XDG_STATE_HOME": "state",
        "TMPDIR": "tmp",
    }.items():
        path = STATE / directory
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    os.environ["DISABLE_UPDATES"] = "1"
    for key in ("OPENAI_API_KEY", "CODEX_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"):
        os.environ.pop(key, None)
    tool_paths = (
        BUNDLE / "runtime" / "capture" / "bin",
        BUNDLE / "runtime" / "bin", CODEX.parent, STATE / "bin",
    )
    os.environ["PATH"] = os.pathsep.join([*(str(path) for path in tool_paths), os.defpath])

    config = STATE / "codex" / "config.toml"
    try:
        with config.open("x", encoding="utf-8") as handle:
            handle.write('cli_auth_credentials_store = "file"\n')
    except FileExistsError:
        pass


def install_claude() -> None:
    if CLAUDE.is_file():
        return
    print("Downloading Claude Code into this folder…", flush=True)
    with urlopen(f"{CLAUDE_RELEASES}/stable", timeout=30) as response:
        version = response.read().decode("ascii").strip()
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version):
        raise ValueError(f"Unexpected Claude Code release version: {version!r}")
    release = f"{CLAUDE_RELEASES}/{version}"
    with urlopen(f"{release}/manifest.json", timeout=30) as response:
        digest = json.load(response)["platforms"]["linux-x64"]["checksum"]
    CLAUDE.parent.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=STATE / "tmp") as directory:
        binary = Path(directory) / "claude"
        with urlopen(f"{release}/linux-x64/claude", timeout=60) as response, binary.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        with binary.open("rb") as handle:
            if hashlib.file_digest(handle, "sha256").hexdigest() != digest:
                raise ValueError("Claude Code download checksum did not match")
        binary.chmod(0o700)
        binary.replace(CLAUDE)


def main() -> int:
    args = sys.argv[1:]
    try:
        setup_environment()
        if args and args[0] in ("login", "logout"):
            if len(args) > 2 or (len(args) == 2 and args[1] not in ("codex", "claude")):
                raise ValueError(f"Usage: ./voice-assistant {args[0]} [codex|claude]")
            auth_command = args[0]
            if args[1:] == ["claude"]:
                if auth_command == "login":
                    install_claude()
                os.execv(CLAUDE, [str(CLAUDE), "auth", auth_command])
            os.execv(CODEX, [str(CODEX), auth_command])

        sys.path.insert(0, str(BUNDLE / "src"))
        from voice_assistant.main import main as run_assistant

        return run_assistant(
            ["--config", str(BUNDLE / "config.toml"), *args],
            agent_executables={"codex": str(CODEX), "claude": str(CLAUDE)},
        )
    except (OSError, ValueError, KeyError) as exc:
        print(f"Portable launcher: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
