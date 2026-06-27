"""
This script is used to visualise the movie data using Streamlit.
This is a multimodal platform that allows users to explore the movie data and get recommendations.
Users can have a global insight of release date, popularity, vote count, vote average, and genre.
Users can search for a movie by title and genre, review the details of a movie, and get similar movies.
Users can also get recommendations for a movie by describing the movie they want to watch.
"""

from ast import literal_eval

import altair as alt
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from chatbot.filters import filter_by_genres, filter_by_year_ranges, get_year_options
from chatbot.formatting import format_genres
from chatbot.llm import DEFAULT_MODEL, get_llm_status
from chatbot.pipeline import recommend_movies
from chatbot.retriever import MovieRetriever
from chatbot.schemas import UserQuery
from config_loader import get_data_path, get_raw_data_path
from logging_config import get_logger, setup_logging
from utils.preprocess import ensure_clean_data

logger = get_logger("app")

DATA_PATH = get_data_path()
RAW_DATA_PATH = get_raw_data_path()
REQUIRED_COLUMNS = (
    "Title",
    "Overview",
    "Release_Date",
    "Popularity",
    "Vote_Count",
    "Vote_Average",
    "Genre",
    "Language",
    "Poster_Url",
)
EXPLORER_DISPLAY_COLUMNS = (
    "Title",
    "Release_Date",
    "Genre",
    "Vote_Average",
    "Vote_Count",
    "Popularity",
    "Language",
)


def get_data_version() -> tuple[float, int, float, tuple[str, ...]]:
    ensure_clean_data(RAW_DATA_PATH, DATA_PATH)
    stat = DATA_PATH.stat()
    raw_mtime = RAW_DATA_PATH.stat().st_mtime if RAW_DATA_PATH.exists() else 0.0
    header = tuple(pd.read_csv(DATA_PATH, nrows=0).columns.tolist())
    return stat.st_mtime, stat.st_size, raw_mtime, header


def validate_dataframe(df: pd.DataFrame, context: str) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(
            f"{context}: missing columns {missing}. "
            "Re-run utils/preprocess.py or update movies.csv and restart the app."
        )
    if df.empty and len(df.columns) == 0:
        raise ValueError(f"{context}: dataframe has no columns. Clear Streamlit cache and reload.")


@st.cache_data
def load_data(data_version: tuple[float, int, float, tuple[str, ...]]) -> pd.DataFrame:
    try:
        df = pd.read_csv(DATA_PATH)
        validate_dataframe(df, "movies_clean.csv")
        df = df.reset_index(drop=True)
        df["Release_Date"] = pd.to_datetime(df["Release_Date"], errors="coerce")
        df["Genre"] = df["Genre"].apply(
            lambda value: literal_eval(value) if isinstance(value, str) and value.startswith("[") else []
        )
        for column in ("Popularity", "Vote_Count", "Vote_Average"):
            df[column] = pd.to_numeric(df[column], errors="coerce")
        logger.info("Loaded %s rows from %s", len(df), DATA_PATH)
        return df
    except Exception:
        logger.exception("Failed to load movie data from %s", DATA_PATH)
        raise


@st.cache_resource
def load_retriever(data_version: tuple[float, int, float, tuple[str, ...]]) -> MovieRetriever:
    df = load_data(data_version)
    return MovieRetriever(df)


@st.cache_data
def build_similarity_matrix(data_version: tuple[float, int, float, tuple[str, ...]]):
    df = load_data(data_version)
    content = (
        df["Title"].fillna("")
        + " "
        + df["Overview"].fillna("")
        + " "
        + df["Genre"].apply(lambda genres: " ".join(genres))
    )
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(content)
    return cosine_similarity(matrix)


def get_similar_movies(df: pd.DataFrame, similarity_matrix, title: str, top_n: int = 5) -> pd.DataFrame:
    matches = df.index[df["Title"] == title].tolist()
    if not matches:
        return pd.DataFrame()
    idx = matches[0]
    scores = list(enumerate(similarity_matrix[idx]))
    scores = sorted(scores, key=lambda item: item[1], reverse=True)[1 : top_n + 1]
    similar_movies = df.iloc[[movie_idx for movie_idx, _ in scores]].copy()
    similar_movies["Similarity"] = [round(score, 3) for _, score in scores]
    return similar_movies


def render_movie_details(movie: pd.Series) -> None:
    detail_col1, detail_col2 = st.columns([1, 2])
    with detail_col1:
        if pd.notna(movie["Poster_Url"]):
            st.image(movie["Poster_Url"], width="stretch")
    with detail_col2:
        st.markdown(f"### {movie['Title']}")
        release_date = movie["Release_Date"]
        release_label = release_date.strftime("%Y-%m-%d") if pd.notna(release_date) else "Unknown"
        st.write(f"**Release date:** {release_label}")
        st.write(f"**Genres:** {format_genres(movie['Genre'])}")
        st.write(f"**Language:** {movie['Language']}")
        st.write(
            f"**Rating:** {movie['Vote_Average']} "
            f"({int(movie['Vote_Count']) if pd.notna(movie['Vote_Count']) else 0} votes)"
        )
        st.write(f"**Popularity:** {movie['Popularity']}")
        st.write(movie["Overview"])


def render_recommendation_cards(movies: pd.DataFrame) -> None:
    for _, movie in movies.iterrows():
        with st.expander(f"{movie['Title']} — rating {movie['Vote_Average']}"):
            card_col1, card_col2 = st.columns([1, 3])
            with card_col1:
                if pd.notna(movie["Poster_Url"]):
                    st.image(movie["Poster_Url"], width="stretch")
            with card_col2:
                st.write(f"**Genres:** {format_genres(movie['Genre'])}")
                st.write(f"**Popularity:** {movie['Popularity']:.0f}")
                if "Match_Score" in movie:
                    st.write(f"**Semantic match:** {movie['Match_Score']}")
                st.write(movie["Overview"])


def run_chatbot_query(
    df: pd.DataFrame,
    retriever: MovieRetriever,
    query: UserQuery,
    user_message: str,
) -> None:
    st.session_state.chat_messages = []

    try:
        with st.spinner("Finding movies..."):
            logger.info(
                "Chatbot query: genres=%s years=%s message=%r",
                query.genres,
                query.year_ranges,
                user_message[:120],
            )
            result = recommend_movies(df, query, retriever)
    except Exception:
        logger.exception("Chatbot query failed")
        st.error("Something went wrong while generating recommendations. See logs/errors.log.")
        return

    st.session_state.chat_messages.append({"role": "user", "content": user_message})
    st.session_state.chat_messages.append(
        {
            "role": "assistant",
            "content": result.message,
            "movies": result.movies,
            "used_llm": result.used_llm,
            "candidate_count": result.candidate_count,
        }
    )


def render_chatbot_tab(df: pd.DataFrame, all_genres: list[str], retriever: MovieRetriever) -> None:
    st.subheader("Movie recommendation chatbot")
    st.caption("Pick at least one genre, optionally refine by year/rating/popularity, then describe what you want to watch.")

    llm_status = get_llm_status()
    if llm_status == "ready":
        st.success(f"Using `{DEFAULT_MODEL}` for chat responses.")
    elif llm_status == "cached":
        logger.info(
            f"Hugging Face model `{DEFAULT_MODEL}` is downloaded."
        )
    else:
        st.error(f"Hugging Face model `{DEFAULT_MODEL}` is not loaded yet.")
        logger.warning(
            f"Hugging Face model `{DEFAULT_MODEL}` is not loaded yet. "
        )

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    with st.form("chatbot_criteria"):
        genres = st.multiselect("Genre *", all_genres, key="chatbot_genres")
        year_ranges = st.multiselect(
            "Release period (optional)",
            get_year_options(df),
            key="chatbot_years",
        )
        col1, col2 = st.columns(2)
        min_rating = col1.slider("Minimum vote average (optional)", 0.0, 10.0, 0.0, 0.1)
        min_popularity = col2.slider("Minimum popularity (optional)", 0.0, float(df["Popularity"].max()), 0.0, 1.0)
        description = st.text_area(
            "Describe the kind of movie you want (optional)",
            placeholder="e.g. slow-burn sci-fi about identity and memory",
        )
        submitted = st.form_submit_button("Get recommendations")

    if submitted:
        if not genres:
            st.error("Please select at least one genre.")
        else:
            query = UserQuery(
                genres=genres,
                year_ranges=year_ranges,
                min_vote_average=min_rating if min_rating > 0 else None,
                min_popularity=min_popularity if min_popularity > 0 else None,
                description=description,
            )
            user_message = description.strip() or f"Recommend {', '.join(genres)} movies."
            run_chatbot_query(df, retriever, query, user_message)

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                if not message.get("used_llm"):
                    st.caption(f"Ranked from {message.get('candidate_count', 0)} matching movies from movie.csv dataset.")
                movies = message.get("movies")
                if movies is not None and not movies.empty:
                    render_recommendation_cards(movies)


def render_explorer_tab(
    df: pd.DataFrame,
    all_genres: list[str],
    similarity_matrix,
) -> None:
    st.subheader("Search")
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    with filter_col1:
        title_query = st.text_input("Search by title", placeholder="e.g. Spider-Man", key="explorer_title")
        selected_genres = st.multiselect("Filter by genre", all_genres, key="explorer_genres")
    with filter_col2:
        year_query = st.multiselect(
            "Filter by release period",
            get_year_options(df),
            key="explorer_years",
        )
        language_query = st.multiselect(
            "Filter by language",
            sorted(df["Language"].dropna().unique()),
            key="explorer_languages",
        )
    with filter_col3:
        min_rating = st.slider("Minimum vote average", 0.0, 10.0, 0.0, 0.1, key="explorer_min_rating")

    filtered = df.copy()
    if title_query:
        filtered = filtered[filtered["Title"].str.contains(title_query, case=False, na=False)]
    if year_query:
        filtered = filter_by_year_ranges(filtered, year_query)
    if language_query:
        filtered = filtered[filtered["Language"].isin(language_query)]
    if selected_genres:
        filtered = filter_by_genres(filtered, selected_genres)
    if min_rating > 0:
        filtered = filtered[filtered["Vote_Average"] >= min_rating]

    if filtered.empty:
        st.warning("No movies found matching your filters. Try changing your search criteria.")
        return

    st.caption("Click a row to view movie details and recommendations.")

    results_table = filtered[list(EXPLORER_DISPLAY_COLUMNS)].head(100).copy().reset_index(drop=True)
    results_table["Genre"] = results_table["Genre"].apply(format_genres)

    selection = st.dataframe(
        results_table,
        on_select="rerun",
        selection_mode="single-row",
        width="stretch",
        key="search_results",
    )

    if selection.selection.rows:
        row_idx = selection.selection.rows[0]
        if 0 <= row_idx < len(results_table):
            st.session_state.selected_movie = results_table.iloc[row_idx]["Title"]
    elif "selected_movie" not in st.session_state:
        st.session_state.selected_movie = results_table.iloc[0]["Title"]

    selected_movie = st.session_state.get("selected_movie")
    movie_row = (
        df.loc[df["Title"] == selected_movie].iloc[0]
        if selected_movie and selected_movie in df["Title"].values
        else None
    )

    st.subheader("Selected movie details")
    if movie_row is not None:
        render_movie_details(movie_row)
    else:
        st.info("Select a movie from the search results.")

    st.subheader("Similar movies")
    similar_movies = (
        get_similar_movies(df, similarity_matrix, selected_movie)
        if movie_row is not None
        else pd.DataFrame()
    )

    if similar_movies.empty:
        st.info("No similar movies found for this title.")
    else:
        for _, similar_movie in similar_movies.iterrows():
            render_movie_details(similar_movie)


def render_global_overview_tab(df: pd.DataFrame, all_genres: list[str]) -> None:
    st.subheader("Global overview")
    chart_col0, chart_col1, chart_col2 = st.columns(3)
    chart_height = 220
    with chart_col0:
        st.metric("Movies", f"{len(df):,}")
        st.metric("Genres", f"{len(all_genres)}")
    with chart_col1:
        st.markdown("**Movies released per year**")
        releases = df.dropna(subset=["Release_Date"]).copy()
        releases["Year"] = releases["Release_Date"].dt.year
        yearly_counts = releases.groupby("Year").size().reset_index(name="Count")
        year_chart = (
            alt.Chart(yearly_counts)
            .mark_bar()
            .encode(x=alt.X("Year:O", title=None), y=alt.Y("Count:Q", title=None))
            .properties(height=chart_height)
        )
        st.altair_chart(year_chart, width="stretch")
    with chart_col2:
        st.markdown("**Popularity vs. rating**")
        scatter_data = df[["Popularity", "Vote_Average"]].dropna()
        scatter_chart = (
            alt.Chart(scatter_data)
            .mark_circle(size=12, opacity=0.35)
            .encode(x=alt.X("Popularity:Q", title=None), y=alt.Y("Vote_Average:Q", title=None))
            .properties(height=chart_height)
        )
        st.altair_chart(scatter_chart, width="stretch")


def main() -> None:
    setup_logging()
    try:
        st.set_page_config(
            page_title="Movie Explorer",
            page_icon="🎬",
            layout="wide",
            initial_sidebar_state="collapsed",
        )
        st.title("🎬 Movie Explorer")
        st.caption("Visualise movie trends, search the catalogue, and get recommendations.")

        data_version = get_data_version()
        df = load_data(data_version)
        retriever = load_retriever(data_version)
        similarity_matrix = build_similarity_matrix(data_version)
        all_genres = sorted({genre for genres in df["Genre"] for genre in genres})
        logger.debug("App started with %s movies and %s genres", len(df), len(all_genres))

        st.markdown(
            """
            <style>
                div[data-testid="stTabs"] button[data-baseweb="tab"] {
                    font-size: 1.35rem;
                    font-weight: 600;
                    padding-top: 0.75rem;
                    padding-bottom: 0.75rem;
                }
                div[data-testid="stTabs"] button[data-baseweb="tab"] p {
                    font-size: 1.35rem;
                    font-weight: 600;
                }
            </style>
            """,
            unsafe_allow_html=True,
        )

        global_overview_tab, explorer_tab, chatbot_tab = st.tabs(
            ["🌍 Global overview", "🔍 Explorer", "🤖 Recommendation chatbot"]
        )
        with global_overview_tab:
            render_global_overview_tab(df, all_genres)
        with explorer_tab:
            render_explorer_tab(df, all_genres, similarity_matrix)
        with chatbot_tab:
            render_chatbot_tab(df, all_genres, retriever)
    except Exception:
        logger.exception("Unhandled error in main")
        st.error("The app encountered an error. Details were written to logs/errors.log.")
        raise


if __name__ == "__main__":
    main()
