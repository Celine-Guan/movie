import numpy as np
import pandas as pd


def _normalize(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    min_value = series.min()
    max_value = series.max()
    if max_value == min_value:
        return pd.Series(1.0, index=series.index)
    return (series - min_value) / (max_value - min_value)


def rank_candidates(
    candidates: pd.DataFrame,
    rag_scores: pd.Series | None = None,
    top_n: int = 5,
) -> pd.DataFrame:
    if candidates.empty:
        return candidates

    ranked = candidates.copy()
    has_description = rag_scores is not None and not rag_scores.empty

    if has_description:
        ranked["_rag"] = rag_scores.reindex(ranked.index).fillna(0.0)
        ranked["_vote"] = _normalize(ranked["Vote_Average"].fillna(0))
        ranked["_popularity"] = _normalize(ranked["Popularity"].fillna(0))
        ranked["_recency"] = _normalize(ranked["Release_Date"].dt.year.fillna(0))
        ranked["_final_score"] = (
            0.6 * ranked["_rag"]
            + 0.2 * ranked["_vote"]
            + 0.1 * ranked["_popularity"]
            + 0.1 * ranked["_recency"]
        )
    else:
        ranked["_vote"] = _normalize(ranked["Vote_Average"].fillna(0))
        ranked["_popularity"] = _normalize(ranked["Popularity"].fillna(0))
        ranked["_recency"] = _normalize(ranked["Release_Date"].dt.year.fillna(0))
        ranked["_final_score"] = (
            0.4 * ranked["_vote"] + 0.3 * ranked["_popularity"] + 0.3 * ranked["_recency"]
        )

    ranked = ranked.sort_values("_final_score", ascending=False).head(top_n)
    if has_description:
        ranked["Match_Score"] = ranked["_rag"].round(3)
    ranked["Final_Score"] = ranked["_final_score"].round(3)
    return ranked.drop(columns=[col for col in ranked.columns if col.startswith("_")])
