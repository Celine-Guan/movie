from __future__ import annotations

import pandas as pd

from chatbot.formatting import format_genres, overview_text
from chatbot.schemas import UserQuery
from config_loader import get_llm_config
from logging_config import get_logger

logger = get_logger("llm")

_llm = get_llm_config()
DEFAULT_MODEL = _llm["model"]
LLM_MAX_NEW_TOKENS = _llm["max_new_tokens"]
LLM_TEMPERATURE = _llm["temperature"]

_generator = None # holds the loaded model in memory 


class HfLocalGenerator:
    def __init__(self, model_name: str) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
        self.model.to(self.device)
        self.model.eval()
        logger.info("Loaded Hugging Face model %s on %s", model_name, self.device)

    def generate(self, prompt: str, max_new_tokens: int, temperature: float) -> str:
        messages = [{"role": "user", "content": prompt}]
        chat_input = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(chat_input, return_tensors="pt").to(self.device)
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if temperature > 0:
            generation_kwargs["temperature"] = temperature

        import torch

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generation_kwargs)

        new_tokens = outputs[0][inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _get_generator(model: str = DEFAULT_MODEL) -> HfLocalGenerator:
    """This is a lazy singleton: the model is loaded only on the first chat request, then reused."""
    global _generator #  the assignment updates the shared module variable
    if _generator is None or _generator.model_name != model:
        _generator = HfLocalGenerator(model)
    return _generator


def is_model_cached(model: str = DEFAULT_MODEL) -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache

        return try_to_load_from_cache(model, "config.json") is not None
    except Exception:
        return False


def get_llm_status(model: str = DEFAULT_MODEL) -> str:
    if _generator is not None and _generator.model_name == model:
        return "ready"
    if is_model_cached(model):
        return "cached"
    return "not_loaded"


def _format_candidate(row: pd.Series, rank: int) -> str:
    release_year = (
        int(row["Release_Date"].year)
        if pd.notna(row["Release_Date"])
        else "Unknown"
    )
    genres = format_genres(row["Genre"])
    overview = overview_text(row["Overview"], limit=240)
    return (
        f"{rank}. {row['Title']} ({release_year}) | "
        f"Rating: {row['Vote_Average']} | Popularity: {row['Popularity']} | "
        f"Genres: {genres} | Overview: {overview}"
    )


# def build_prompt(query: UserQuery, movies: pd.DataFrame) -> str:
#     criteria = [f"Genres: {', '.join(query.genres)}"]
#     if query.year_ranges:
#         criteria.append(f"Release periods: {', '.join(query.year_ranges)}")
#     if query.min_vote_average:
#         criteria.append(f"Minimum rating: {query.min_vote_average}")
#     if query.min_popularity:
#         criteria.append(f"Minimum popularity: {query.min_popularity}")
#     if query.semantic_text():
#         criteria.append(f"User description: {query.semantic_text()}")

#     candidate_lines = [_format_candidate(row, index + 1) for index, (_, row) in enumerate(movies.iterrows())]
#     return f"""You are a helpful movie recommendation assistant.
# Recommend ONLY movies from the candidate list below.
# Do not invent titles.
# Return 3 to 5 picks as a short friendly chat response.
# For each pick, explain briefly why it matches the user's criteria using the overview/genres.

# User criteria:
# - {"\n".join(criteria)}

# Candidate movies:
# {"\n".join(candidate_lines)}
# """

def build_prompt(query: UserQuery, movies: pd.DataFrame) -> str:
    criteria = [f"Genres: {', '.join(query.genres)}"]
    if query.year_ranges:
        criteria.append(f"Release periods: {', '.join(query.year_ranges)}")
    if query.min_vote_average:
        criteria.append(f"Minimum rating: {query.min_vote_average}")
    if query.min_popularity:
        criteria.append(f"Minimum popularity: {query.min_popularity}")
    if query.semantic_text():
        criteria.append(f"User description: {query.semantic_text()}")

    candidate_lines = [_format_candidate(row, index + 1) for index, (_, row) in enumerate(movies.iterrows())]
    criteria_text = "\n".join(criteria)
    candidates_text = "\n".join(candidate_lines)
    return f"""You are a helpful movie recommendation assistant.
Recommend ONLY movies from the candidate list below.
Do not invent titles.
Return 3 to 5 picks as a short friendly chat response.
For each pick, explain briefly why it matches the user's criteria using the overview/genres.

User criteria:
- {criteria_text}

Candidate movies:
{candidates_text}
"""


def generate_response(
    query: UserQuery,
    movies: pd.DataFrame,
    model: str = DEFAULT_MODEL,
) -> tuple[str, bool]:
    if movies.empty:
        return (
            "I couldn't find movies that match your filters. "
            "Try relaxing the year, rating, or popularity constraints.",
            False,
        )

    prompt = build_prompt(query, movies)
    try:
        generator = _get_generator(model)
        response = generator.generate(
            prompt,
            max_new_tokens=LLM_MAX_NEW_TOKENS,
            temperature=LLM_TEMPERATURE,
        )
        if response:
            return response, True
        logger.warning("Hugging Face model returned an empty response; using fallback")
    except Exception:
        logger.warning("Hugging Face generation failed; using fallback response", exc_info=True)

    return fallback_response(query, movies), False


def fallback_response(query: UserQuery, movies: pd.DataFrame) -> str:
    lines = [
        "Here are my top picks based on your filters"
        + (f" and description: **{query.semantic_text()}**" if query.semantic_text() else "")
        + ":",
        "",
    ]
    for _, row in movies.iterrows():
        genres = format_genres(row["Genre"])
        reason = overview_text(row["Overview"], limit=180)
        lines.append(f"**{row['Title']}** ({genres}) — rating {row['Vote_Average']}, popularity {row['Popularity']:.0f}.")
        lines.append(reason)
        lines.append("")
    return "\n".join(lines)
