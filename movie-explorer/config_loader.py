from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

APP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = APP_DIR / "config.yaml"


@lru_cache
def load_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")
    with CONFIG_PATH.open(encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def _resolve_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (APP_DIR / path).resolve()


def get_raw_data_path() -> Path:
    return _resolve_path(load_config()["data"]["raw_movies_csv"])


def get_data_path() -> Path:
    return _resolve_path(load_config()["data"]["movies_csv"])


def get_llm_config() -> dict[str, Any]:
    return load_config()["llm"]


def get_rag_config() -> dict[str, Any]:
    return load_config()["rag"]


def get_recommendations_config() -> dict[str, Any]:
    return load_config()["recommendations"]


def get_logging_config() -> dict[str, Any]:
    return load_config()["logging"]
