#!/usr/bin/env python3
"""Create venv, install dependencies, and build movies_clean.csv."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
APP_DIR = REPO_ROOT / "movie-explorer"
VENV_DIR = REPO_ROOT / ".venv"
REQUIREMENTS = APP_DIR / "requirements.txt"
MIN_PYTHON = (3, 11)


def python_version(executable: str) -> tuple[int, int] | None:
    try:
        output = subprocess.check_output(
            [executable, "-c", "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        major, minor = output.split(".")
        return int(major), int(minor)
    except (subprocess.CalledProcessError, OSError, ValueError):
        return None


def find_python() -> str | None:
    candidates: list[str] = []

    def add(name: str) -> None:
        path = shutil.which(name)
        if path and path not in candidates:
            candidates.append(path)

    add(sys.executable)
    for minor in range(MIN_PYTHON[1], MIN_PYTHON[1] + 6):
        add(f"python3.{minor}")
        add(f"python{MIN_PYTHON[0]}.{minor}")
    add("python3")

    for executable in candidates:
        version = python_version(executable)
        if version and version >= MIN_PYTHON:
            return executable
    return None


def print_python_install_help() -> None:
    print(f"\nPython {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ is required (you are running {sys.version.split()[0]}).")
    print("Install Python 3.11+ using one of the options below, then re-run:\n")
    print("  python3 setup.py\n")
    if sys.platform == "darwin":
        print("  macOS (Homebrew):  brew install python@3.11")
        print("  pyenv:             pyenv install 3.11 && pyenv local 3.11")
    elif sys.platform.startswith("linux"):
        print("  Ubuntu/Debian:     sudo apt update && sudo apt install python3.11 python3.11-venv")
        print("  pyenv:             pyenv install 3.11 && pyenv local 3.11")
    elif sys.platform == "win32":
        print("  Windows:           winget install Python.Python.3.11")
        print("                     https://www.python.org/downloads/")
    else:
        print("  https://www.python.org/downloads/")
    print("\nIf Python 3.11+ is already installed, try explicitly:")
    print("  python3.11 setup.py")


def venv_python() -> Path:
    if sys.platform == "win32":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def run(command: list[str], *, cwd: Path = APP_DIR) -> None:
    print(f"$ {' '.join(command)}")
    subprocess.check_call(command, cwd=cwd)


def main() -> int:
    python_executable = find_python()
    if python_executable is None:
        print_python_install_help()
        return 1

    if python_executable != sys.executable:
        version = python_version(python_executable)
        print(f"Using {python_executable} (Python {version[0]}.{version[1]})")

    if not REQUIREMENTS.exists():
        print(f"Missing requirements file: {REQUIREMENTS}")
        return 1

    if not VENV_DIR.exists():
        run([python_executable, "-m", "venv", str(VENV_DIR)], cwd=REPO_ROOT)
    else:
        print(f"Using existing virtual environment at {VENV_DIR}")

    python = venv_python()
    run([str(python), "-m", "pip", "install", "--upgrade", "pip"])
    run([str(python), "-m", "pip", "install", "-r", str(REQUIREMENTS)])

    sys.path.insert(0, str(APP_DIR))
    from config_loader import get_data_path, get_raw_data_path

    raw_csv = get_raw_data_path()
    clean_csv = get_data_path()
    if not raw_csv.exists():
        print(f"\nWarning: raw data not found at {raw_csv}")
        print("Place movies.csv there or update data paths in config.yaml.")
        if not clean_csv.exists():
            print("movies_clean.csv is also missing — the app will fail until data is available.")
            return 1
        print(f"Found existing {clean_csv}; skipping preprocessing.")
    else:
        run([str(python), "-m", "utils.preprocess"])

    activate = ".venv\\Scripts\\activate" if sys.platform == "win32" else "source .venv/bin/activate"
    print("\nSetup complete.")
    print(f"  {activate}")
    print(f"  cd {APP_DIR.name}")
    print("  streamlit run app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
