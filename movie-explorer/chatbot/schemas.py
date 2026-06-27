from dataclasses import dataclass, field

import pandas as pd


@dataclass
class UserQuery:
    genres: list[str]
    year_ranges: list[str] = field(default_factory=list)
    min_vote_average: float | None = None
    min_popularity: float | None = None
    description: str = ""

    def semantic_text(self) -> str:
        return self.description.strip()


@dataclass
class RecommendationResult:
    movies: pd.DataFrame
    message: str
    used_llm: bool
    candidate_count: int
