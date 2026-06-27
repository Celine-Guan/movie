#!/usr/bin/env python3
"""Create venv, install dependencies, and build movies_clean.csv."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
VENV_DIR = PROJECT_DIR / ".venv"
REQUIREMENTS = PROJECT_DIR / "requirements.txt"
MIN_PYTHON = (3, 11)


def venv_python() -> Path:
    if sys.platform == "win32":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def run(command: list[str], *, cwd: Path = PROJECT_DIR) -> None:
    print(f"$ {' '.join(command)}")
    subprocess.check_call(command, cwd=cwd)


def main() -> int:
    if sys.version_info[:2] < MIN_PYTHON:
        print(f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ is required.")
        return 1

    if not VENV_DIR.exists():
        run([sys.executable, "-m", "venv", str(VENV_DIR)])
    else:
        print(f"Using existing virtual environment at {VENV_DIR}")

    python = venv_python()
    run([str(python), "-m", "pip", "install", "--upgrade", "pip"])
    run([str(python), "-m", "pip", "install", "-r", str(REQUIREMENTS)])

    sys.path.insert(0, str(PROJECT_DIR))
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
    print(f"  cd {PROJECT_DIR.name}")
    print(f"  {activate}")
    print("  streamlit run app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
