"""Preprocess raw movie CSV into a cleaned dataset."""

import logging
from pathlib import Path

import pandas as pd
import pycountry

logger = logging.getLogger(__name__)


def clean_genres(genre) -> list[str]:
    if pd.isna(genre):
        return []
    return sorted({g.strip() for g in str(genre).split(",") if g.strip()})


def code_to_language(code) -> str:
    if pd.isna(code) or str(code).strip() == "":
        return "Unknown language"
    code = str(code).strip().lower()
    if code == "cn":
        return "Chinese"
    if len(code) != 2:
        return "Unknown language"
    lang = pycountry.languages.get(alpha_2=code)
    return lang.name if lang else code.upper()


def preprocess_movies(raw_path: Path, output_path: Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(raw_path, engine="python")

    df = df[df.notna().any(axis=1)]
    df = df[df["Title"].notna()]
    df["Title"] = df["Title"].str.strip()

    df["Release_Date"] = pd.to_datetime(df["Release_Date"], errors="coerce")
    df["Release_Date"] = df["Release_Date"].fillna(0)

    df["Overview"] = df["Overview"].str.strip()
    df["Overview"] = df["Overview"].fillna("")

    df["Genre"] = df["Genre"].apply(clean_genres)
    df["Language"] = df["Original_Language"].apply(code_to_language)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info("Saved %s rows to %s", len(df), output_path)

    return df


def ensure_clean_data(raw_path: Path, clean_path: Path) -> Path:
    if not raw_path.exists():
        if clean_path.exists():
            logger.warning("Raw data not found at %s; using existing clean file", raw_path)
            return clean_path
        raise FileNotFoundError(f"Raw movie data not found: {raw_path}")

    needs_preprocessing = (
        not clean_path.exists() or raw_path.stat().st_mtime > clean_path.stat().st_mtime
    )
    if needs_preprocessing:
        logger.info("Preprocessing %s -> %s", raw_path, clean_path)
        preprocess_movies(raw_path, clean_path)

    return clean_path


if __name__ == "__main__":
    from config_loader import get_data_path, get_raw_data_path
    from logging_config import setup_logging

    setup_logging()
    preprocess_movies(get_raw_data_path(), get_data_path())
