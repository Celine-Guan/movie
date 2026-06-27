import pandas as pd

from chatbot.ranker import rank_candidates


def test_rank_candidates_without_rag_uses_metadata_scores(sample_movies):
    ranked = rank_candidates(sample_movies, rag_scores=None, top_n=2)
    assert len(ranked) == 2
    assert ranked.iloc[0]["Vote_Average"] >= ranked.iloc[1]["Vote_Average"]


def test_rank_candidates_with_rag_prioritizes_semantic_match(sample_movies):
    rag_scores = pd.Series({0: 0.2, 1: 0.9, 2: 0.5, 3: 0.1})
    ranked = rank_candidates(sample_movies, rag_scores=rag_scores, top_n=1)
    assert ranked.iloc[0]["Title"] == "Film B"
    assert ranked.iloc[0]["Match_Score"] == 0.9


def test_rank_candidates_returns_empty_for_no_candidates():
    empty = pd.DataFrame(columns=["Vote_Average", "Popularity", "Release_Date"])
    assert rank_candidates(empty).empty
