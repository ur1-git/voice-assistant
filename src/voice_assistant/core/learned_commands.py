"""Learn a voice shortcut for an installed desktop application."""
from collections.abc import Iterator
from configparser import ConfigParser, Error as ConfigParserError
from contextlib import closing
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import sys

from .commands import CommandError, LocalCommand, normalize_activation


OPEN_REQUEST = re.compile(
    r"(?:please |(?:can|could|would) you (?:please )?)?(?:open|launch|start) (.+?)(?: please)?"
)
LAUNCH_TOOL = "LAUNCH_DESKTOP_APP "


@dataclass(frozen=True)
class DesktopApp:
    name: str
    path: Path

    def command(self, launcher: str) -> LocalCommand:
        # Desktop apps may inherit the launcher's output streams and outlive it.
        return LocalCommand(
            (launcher, "launch", str(self.path)), f"Opening {self.name}.", capture_output=False,
        )


def desktop_entries() -> Iterator[tuple[str, Path]]:
    home = Path.home()
    data_home = os.environ.get("XDG_DATA_HOME") or str(home / ".local/share")
    data_dirs = os.environ.get("XDG_DATA_DIRS") or "/usr/local/share:/usr/share"
    seen = set()
    for root in (
        data_home, *data_dirs.split(os.pathsep),
        str(home / ".local/share/flatpak/exports/share"), "/var/lib/flatpak/exports/share",
    ):
        if not Path(root).is_absolute():
            continue
        applications = Path(root) / "applications"
        for path in applications.rglob("*.desktop"):
            desktop_id = path.relative_to(applications).as_posix().replace("/", "-")
            if desktop_id not in seen:
                seen.add(desktop_id)
                yield desktop_id, path


def read_desktop_app(path: Path) -> DesktopApp | None:
    parser = ConfigParser(interpolation=None, strict=False, delimiters=("=",))
    try:
        with path.open(encoding="utf-8") as handle:
            parser.read_file(handle)
    except (OSError, UnicodeError, ConfigParserError):
        return None
    if not parser.has_section("Desktop Entry"):
        return None
    entry = parser["Desktop Entry"]
    name = entry.get("Name", "").strip()
    if not name or entry.get("Type") != "Application":
        return None
    if entry.get("Hidden") == "true" or entry.get("NoDisplay") == "true":
        return None
    if not entry.get("Exec") and entry.get("DBusActivatable") != "true":
        return None
    executable = entry.get("TryExec")
    if executable and shutil.which(executable) is None:
        return None
    return DesktopApp(name, path)


def name_key(name: str) -> str:
    return " ".join(re.findall(r"[^\W_]+", name.casefold()))


class LearnedCommands:
    def __init__(self, config_path: Path):
        state_home = Path(os.environ.get("XDG_STATE_HOME") or Path.home() / ".local/state")
        config_key = sha256(os.fsencode(config_path.expanduser().resolve())).hexdigest()[:16]
        self.path = state_home / "voice-assistant" / f"{config_key}.sqlite3"

    def lookup(self, phrase: str) -> LocalCommand | None:
        key = normalize_activation(phrase)
        if OPEN_REQUEST.fullmatch(key) is None or not self.path.is_file():
            return None
        try:
            with closing(sqlite3.connect(self.path)) as database:
                row = database.execute("SELECT desktop_id FROM shortcuts WHERE phrase = ?", (key,)).fetchone()
        except sqlite3.Error as exc:
            print(f"Could not read learned voice shortcuts: {exc}", file=sys.stderr)
            return None
        if row is not None and (launcher := shutil.which("gio")) is not None:
            for desktop_id, path in desktop_entries():
                if desktop_id == row[0]:
                    app = read_desktop_app(path)
                    return app.command(launcher) if app is not None else None
        return None

    def candidates(self, phrase: str) -> dict[str, DesktopApp]:
        request = OPEN_REQUEST.fullmatch(normalize_activation(phrase))
        if request is None or shutil.which("gio") is None:
            return {}
        target = name_key(request[1])
        candidates = {}
        for desktop_id, path in desktop_entries():
            app = read_desktop_app(path)
            if app is not None and target in (
                name_key(app.name), name_key(app.name.split(" (", 1)[0]),
                name_key(desktop_id.removesuffix(".desktop").rsplit(".", 1)[-1]),
            ):
                candidates[desktop_id] = app
        return candidates

    @staticmethod
    def instructions(candidates: dict[str, DesktopApp]) -> str:
        apps = {desktop_id: app.name for desktop_id, app in candidates.items()}
        return (
            "The voice assistant can open an installed app now and learn this exact phrase as a shortcut. "
            "If the current request is to open one of these apps, call the tool by replying with one line: "
            f"{LAUNCH_TOOL}<desktop_id>. Substitute a listed ID and omit the brackets and final period. "
            "The voice assistant will perform the launch and save the shortcut. Let it handle both steps. "
            "If multiple apps fit equally well, ask which one. Otherwise respond normally for requests "
            f"that are not app launches. Available desktop IDs and their app names: {json.dumps(apps)}"
        )

    def finish(
        self, phrase: str, response: str, candidates: dict[str, DesktopApp], directory: Path, timeout: int,
    ) -> str:
        if not response.startswith(LAUNCH_TOOL):
            return response
        desktop_id = response.removeprefix(LAUNCH_TOOL).strip()
        app = candidates.get(desktop_id)
        launcher = shutil.which("gio")
        if app is None or launcher is None:
            return "I couldn't match that request to an installed app."
        command = app.command(launcher)
        try:
            command.run(directory, timeout)
        except CommandError as exc:
            return f"I couldn't open {app.name}. {exc}"
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with closing(sqlite3.connect(self.path)) as database, database:
                self._create_table(database)
                database.execute(
                    "INSERT INTO shortcuts (phrase, desktop_id) VALUES (?, ?) "
                    "ON CONFLICT(phrase) DO UPDATE SET desktop_id = excluded.desktop_id",
                    (normalize_activation(phrase), desktop_id),
                )
        except (OSError, sqlite3.Error):
            return f"{command.reply} I couldn't save the voice shortcut."
        return command.reply

    @staticmethod
    def _create_table(database: sqlite3.Connection) -> None:
        database.execute("CREATE TABLE IF NOT EXISTS shortcuts (phrase TEXT PRIMARY KEY, desktop_id TEXT NOT NULL)")
