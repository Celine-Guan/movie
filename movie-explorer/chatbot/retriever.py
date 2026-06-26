import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from config_loader import get_rag_config
from logging_config import get_logger

logger = get_logger("retriever")

_rag = get_rag_config()
DEFAULT_EMBEDDING_MODEL = _rag["embedding_model"]
EMBEDDING_BATCH_SIZE = _rag["batch_size"]


def _movie_document(row: pd.Series) -> str:
    genres = ", ".join(row["Genre"]) if row["Genre"] else "Unknown"
    overview = row["Overview"] if isinstance(row["Overview"], str) else ""
    return f"{row['Title']}. Genres: {genres}. {overview}"


class MovieRetriever:
    def __init__(self, df: pd.DataFrame, model_name: str = DEFAULT_EMBEDDING_MODEL):
        self.df = df.reset_index(drop=True)
        self.model = SentenceTransformer(model_name)
        documents = [_movie_document(row) for _, row in self.df.iterrows()]
        self.embeddings = self.model.encode(
            documents,
            batch_size=EMBEDDING_BATCH_SIZE,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        logger.info("Built embeddings for %s movies using %s", len(self.df), model_name)

    def search(
        self,
        query: str,
        candidate_indices: list[int],
        top_k: int = 20,
    ) -> pd.Series:
        if not query.strip() or not candidate_indices:
            return pd.Series(dtype=float)

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
        )[0]
        subset_embeddings = self.embeddings[candidate_indices]
        scores = subset_embeddings @ query_embedding
        top_positions = np.argsort(scores)[::-1][:top_k]

        result_index = [candidate_indices[position] for position in top_positions]
        result_scores = [float(scores[position]) for position in top_positions]
        return pd.Series(result_scores, index=result_index)
