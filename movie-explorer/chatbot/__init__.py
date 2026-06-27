from chatbot.schemas import RecommendationResult, UserQuery

__all__ = ["UserQuery", "RecommendationResult", "recommend_movies"]


def __getattr__(name: str):
    if name == "recommend_movies":
        from chatbot.pipeline import recommend_movies

        return recommend_movies
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
