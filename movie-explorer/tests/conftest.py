import pandas as pd
import pytest


@pytest.fixture
def sample_movies() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Title": ["Film A", "Film B", "Film C", "Film D"],
            "Overview": ["A space adventure.", "A romantic comedy.", "A dark thriller.", "An old classic."],
            "Release_Date": pd.to_datetime(["2020-01-01", "2015-06-15", "1990-03-20", "1945-12-01"]),
            "Popularity": [80.0, 40.0, 60.0, 10.0],
            "Vote_Average": [8.0, 6.5, 7.0, 9.0],
            "Genre": [
                ["Action", "Science Fiction"],
                ["Romance", "Comedy"],
                ["Thriller"],
                ["Drama"],
            ],
            "Language": ["English", "French", "English", "English"],
        }
    )
