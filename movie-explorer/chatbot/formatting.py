def format_genres(genres) -> str:
    return ", ".join(genres) if genres else "Unknown"


def overview_text(overview, *, limit: int | None = None) -> str:
    text = overview if isinstance(overview, str) else ""
    if limit is not None and len(text) > limit:
        return text[:limit] + "..."
    return text
