import pandas as pd

YEAR_RANGES: list[tuple[str, int | None, int | None]] = [
    ("Before 1950", None, 1949),
    ("1950-1960", 1950, 1960),
    ("1960-1969", 1960, 1969),
    ("1970-1979", 1970, 1979),
    ("1980-1989", 1980, 1989),
    ("1990-1999", 1990, 1999),
    ("2000-2009", 2000, 2009),
    ("2010-2019", 2010, 2019),
    ("2020-present", 2020, None),
]


def _year_range_mask(years: pd.Series, start: int | None, end: int | None) -> pd.Series:
    if start is None:
        return years <= end
    if end is None:
        return years >= start
    return (years >= start) & (years <= end)


def get_year_options(df: pd.DataFrame) -> list[str]:
    years = df["Release_Date"].dt.year
    return [
        label
        for label, start, end in YEAR_RANGES
        if _year_range_mask(years, start, end).any()
    ]


def filter_by_year_ranges(df: pd.DataFrame, selected_ranges: list[str]) -> pd.DataFrame:
    if not selected_ranges:
        return df
    years = df["Release_Date"].dt.year
    mask = pd.Series(False, index=df.index)
    for label, start, end in YEAR_RANGES:
        if label in selected_ranges:
            mask |= _year_range_mask(years, start, end)
    return df[mask]


def filter_by_genres(df: pd.DataFrame, genres: list[str]) -> pd.DataFrame:
    if not genres:
        return df
    return df[df["Genre"].apply(lambda movie_genres: any(genre in movie_genres for genre in genres))]


def apply_metadata_filters(
    df: pd.DataFrame,
    genres: list[str],
    year_ranges: list[str] | None = None,
    min_vote_average: float | None = None,
    min_popularity: float | None = None,
) -> pd.DataFrame:
    if not genres:
        return df.iloc[0:0]

    filtered = filter_by_genres(df, genres)

    if year_ranges:
        filtered = filter_by_year_ranges(filtered, year_ranges)
    if min_vote_average is not None and min_vote_average > 0:
        filtered = filtered[filtered["Vote_Average"] >= min_vote_average]
    if min_popularity is not None and min_popularity > 0:
        filtered = filtered[filtered["Popularity"] >= min_popularity]

    return filtered
