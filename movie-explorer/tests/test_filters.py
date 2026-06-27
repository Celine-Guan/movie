import pandas as pd

from chatbot.filters import apply_metadata_filters, filter_by_genres, filter_by_year_ranges


def test_filter_by_genres_returns_matching_movies(sample_movies):
    result = filter_by_genres(sample_movies, ["Romance"])
    assert list(result["Title"]) == ["Film B"]


def test_filter_by_year_ranges(sample_movies):
    result = filter_by_year_ranges(sample_movies, ["2020-present"])
    assert list(result["Title"]) == ["Film A"]


def test_apply_metadata_filters_requires_genres(sample_movies):
    result = apply_metadata_filters(sample_movies, genres=[])
    assert result.empty


def test_apply_metadata_filters_combines_constraints(sample_movies):
    result = apply_metadata_filters(
        sample_movies,
        genres=["Action", "Science Fiction"],
        min_vote_average=7.5,
    )
    assert list(result["Title"]) == ["Film A"]
