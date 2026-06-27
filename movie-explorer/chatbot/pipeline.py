import pandas as pd

from chatbot.filters import apply_metadata_filters
from chatbot.llm import DEFAULT_MODEL, generate_response
from chatbot.ranker import rank_candidates
from chatbot.retriever import MovieRetriever
from chatbot.schemas import RecommendationResult, UserQuery
from config_loader import get_recommendations_config
from logging_config import get_logger

logger = get_logger("pipeline")

_rec = get_recommendations_config()
DEFAULT_TOP_N = _rec["top_n"]
RAG_TOP_K_MULTIPLIER = _rec["rag_top_k_multiplier"]
RAG_TOP_K_MIN = _rec["rag_top_k_min"]


def recommend_movies(
    df: pd.DataFrame,
    query: UserQuery,
    retriever: MovieRetriever,
    top_n: int = DEFAULT_TOP_N,
    llm_model: str = DEFAULT_MODEL,
) -> RecommendationResult:
    candidates = apply_metadata_filters(
        df=df,
        genres=query.genres,
        year_ranges=query.year_ranges,
        min_vote_average=query.min_vote_average,
        min_popularity=query.min_popularity,
    )

    if candidates.empty:
        logger.info("No candidates after metadata filters for genres=%s", query.genres)
        return RecommendationResult(
            movies=candidates,
            message="No movies match your filters. Try selecting different genres or relaxing the optional constraints.",
            used_llm=False,
            candidate_count=0,
        )

    semantic_text = query.semantic_text()  # user's description, stripped
    rag_scores = None
    if semantic_text:
        candidate_indices = candidates.index.tolist()
        rag_scores = retriever.search(
            semantic_text,
            candidate_indices,
            top_k=max(top_n * RAG_TOP_K_MULTIPLIER, RAG_TOP_K_MIN),
        )

    ranked = rank_candidates(candidates, rag_scores=rag_scores, top_n=top_n)
    message, used_llm = generate_response(query, ranked, model=llm_model)
    logger.info(
        "Recommendations ready: candidates=%s top_n=%s used_llm=%s",
        len(candidates),
        len(ranked),
        used_llm,
    )

    return RecommendationResult(
        movies=ranked,
        message=message,
        used_llm=used_llm,
        candidate_count=len(candidates),
    )
