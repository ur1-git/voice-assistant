"""Build the Linux x86-64 distribution from pinned upstream releases."""
import argparse
import hashlib
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory
from urllib.request import urlopen


PROJECT = Path(__file__).resolve().parents[1]
NAME = "voice-assistant-linux-x86_64"
PYTHON_RELEASE = "20261110"
PYTHON_ARCHIVE = f"cpython-3.13.18+{PYTHON_RELEASE}-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz"
CODEX_TAG = "rust-v0.170.0"
CODEX_ARCHIVE = "codex-package-x86_64-unknown-linux-musl.tar.gz"
WHISPER_TAG = "v1.9.5"
MODEL = "ggml-base.bin"


def verify(path: Path, algorithm: str, expected: str) -> None:
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, algorithm).hexdigest()
    if digest != expected:
        raise ValueError(f"Checksum mismatch for {path}. Supply a fresh copy of that download.")


def fetch(cache: Path, name: str, url: str, checksum: tuple[str, str] | None = None) -> Path:
    path = cache / name
    if path.is_file():
        if checksum:
            verify(path, *checksum)
        return path

    print(f"Downloading {name}", flush=True)
    with TemporaryDirectory(dir=cache) as directory:
        partial = Path(directory) / name
        try:
            with urlopen(url, timeout=45) as response, partial.open("wb") as handle:
                shutil.copyfileobj(response, handle)
        except OSError as exc:
            raise OSError(f"Could not fetch {url}. Provide {name} in {cache}. {exc}") from exc
        if checksum:
            verify(partial, *checksum)
        partial.replace(path)
    return path


def unpack(archive: Path, target: Path) -> None:
    with tarfile.open(archive) as package:
        package.extractall(target, filter="data")


def build_whisper(source: Path, build: Path) -> Path:
    subprocess.run([
        "cmake", "-S", str(source), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release",
        "-DBUILD_SHARED_LIBS=OFF", "-DWHISPER_BUILD_TESTS=OFF", "-DWHISPER_BUILD_SERVER=OFF",
        "-DGGML_NATIVE=OFF",
        "-DGGML_AVX=OFF", "-DGGML_AVX2=OFF", "-DGGML_AVX512=OFF", "-DGGML_FMA=OFF",
        "-DGGML_F16C=OFF", "-DGGML_BMI2=OFF", "-DGGML_SSE42=OFF", "-DGGML_OPENMP=OFF",
        "-DCMAKE_EXE_LINKER_FLAGS=-static-libstdc++ -static-libgcc",
    ], check=True)
    subprocess.run(["cmake", "--build", str(build), "--target", "whisper-cli", "--parallel", "4"], check=True)
    return build / "bin" / "whisper-cli"


def assemble(bundle: Path, work: Path, assets: Path, capture: Path | None) -> None:
    runtime = bundle / "runtime"
    tools = runtime / "bin"
    licenses = bundle / "licenses"
    models = bundle / "models"
    for directory in (tools, licenses, models, bundle / "workspace"):
        directory.mkdir(parents=True)

    python = fetch(
        assets, PYTHON_ARCHIVE,
        f"https://github.com/astral-sh/python-build-standalone/releases/download/{PYTHON_RELEASE}/{PYTHON_ARCHIVE}",
        ("sha256", "f2d2199eb25880e07863b391cf7124e7ffa1338b0d65600dfcec117c4223a7fb"),
    )
    unpack(python, runtime)

    codex = fetch(
        assets, CODEX_ARCHIVE, f"https://github.com/openai/codex/releases/download/{CODEX_TAG}/{CODEX_ARCHIVE}",
        ("sha256", "a77ff87be7ba809d715728270a78f5ce3527dc07529416d1ef8929b90b8475eb"),
    )
    unpack(codex, work / "codex")
    (codex_metadata,) = (work / "codex").rglob("codex-package.json")
    codex_metadata.parent.rename(runtime / "codex")
    for name in ("LICENSE", "NOTICE"):
        document = fetch(
            assets, f"codex-{CODEX_TAG}-{name}",
            f"https://raw.githubusercontent.com/openai/codex/{CODEX_TAG}/{name}",
        )
        shutil.copyfile(document, licenses / f"Codex-{name}")

    whisper = fetch(
        assets, f"whisper.cpp-{WHISPER_TAG}.tar.gz",
        f"https://github.com/ggml-org/whisper.cpp/archive/refs/tags/{WHISPER_TAG}.tar.gz",
    )
    unpack(whisper, work / "whisper")
    source = work / "whisper" / f"whisper.cpp-{WHISPER_TAG.removeprefix('v')}"
    shutil.copy2(build_whisper(source, work / "build-whisper"), tools / "whisper-cli")
    shutil.copyfile(source / "LICENSE", licenses / "Whisper-LICENSE")
    model = fetch(
        assets, MODEL, f"https://huggingface.co/ggerganov/whisper.cpp/resolve/main/{MODEL}",
        ("sha1", "465707469ff3a37a2b9b8d8f89f2f99de7299dac"),
    )
    shutil.copyfile(model, models / MODEL)
    if capture is not None:
        shutil.copytree(capture, runtime / "capture", symlinks=True)

    shutil.copytree(PROJECT / "src", bundle / "src", ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
    shutil.copyfile(PROJECT / "README.md", bundle / "README.md")
    shutil.copyfile(PROJECT / "portable" / "launch.py", bundle / "launch.py")
    launcher = bundle / "voice-assistant"
    shutil.copyfile(PROJECT / "portable" / "voice-assistant", launcher)
    launcher.chmod(0o755)
    config = (PROJECT / "config.example.toml").read_text(encoding="utf-8")
    config = config.replace('working_directory = "."', 'working_directory = "workspace"')
    config = config.replace("models/ggml-small.bin", f"models/{MODEL}")
    (bundle / "config.toml").write_text(config, encoding="utf-8")

    # Test the runtime and native binaries before publishing an archive.
    commands = [
        [launcher, "--help"],
        [runtime / "codex" / "bin" / "codex", "--version"],
        [tools / "whisper-cli", "--version"],
    ]
    if capture is not None:
        commands.append([runtime / "capture" / "bin" / "pw-record", "--version"])
    smoke_env = dict(os.environ, CODEX_HOME=str(bundle / ".state" / "codex"))
    for command in commands:
        subprocess.run(command, check=True, cwd=bundle, stdout=subprocess.DEVNULL, env=smoke_env)
    shutil.rmtree(bundle / ".state")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT / "dist", help="Directory for the finished archive")
    parser.add_argument("--assets", type=Path, help="Directory for cached upstream downloads")
    parser.add_argument("--capture", type=Path, help="Optional relocatable recording prefix with bin/pw-record")
    args = parser.parse_args()
    if sys.platform != "linux" or platform.machine() != "x86_64":
        parser.error("The bundle builder requires an x86-64 Linux build machine")
    if args.capture is not None and not (args.capture / "bin" / "pw-record").is_file():
        parser.error("--capture must contain bin/pw-record")
    if args.capture is not None:
        args.capture = args.capture.resolve()
        for path in args.capture.rglob("*"):
            if path.is_symlink() and (path.readlink().is_absolute() or not path.resolve().is_relative_to(args.capture)):
                parser.error(f"Recording prefix symlinks must stay inside the prefix: {path}")
    output = args.output.resolve()
    assets = (args.assets or output / ".downloads").resolve()
    try:
        output.mkdir(parents=True, exist_ok=True)
        assets.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=output) as directory:
            work = Path(directory)
            bundle = work / NAME
            assemble(bundle, work, assets, args.capture)
            archive = work / f"{NAME}.tar.gz"
            with tarfile.open(archive, "w:gz", compresslevel=3) as package:
                package.add(bundle, arcname=NAME)
            destination = archive.replace(output / archive.name)
        print(destination)
    except (OSError, ValueError, tarfile.TarError, subprocess.CalledProcessError) as exc:
        print(f"Portable build failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
