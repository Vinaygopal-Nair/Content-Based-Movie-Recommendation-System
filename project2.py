"""
Movie recommender backend (Phase 1) - TMDB 5000 dataset.

Pipeline:
    load_data -> preprocess_data -> build_features -> build_similarity_matrix
    -> find_movie -> recommend

Usage:
    python project2.py Inception          # one-off query
    python project2.py                    # prompts: "Enter movie name:"
    python project2.py --check            # run sanity checks
    python project2.py --panel            # print recs for a fixed set of seed movies
"""
import argparse
import ast
import re
import os
from difflib import get_close_matches
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------------------------------
# Settings (change these, not the code below)
# --------------------------------------------------------------------------
DATA_DIR = Path(os.environ.get("TMDB_DATA_DIR", "data"))
MOVIES_CSV = DATA_DIR / "tmdb_5000_movies.csv"
CREDITS_CSV = DATA_DIR / "tmdb_5000_credits.csv"

TOP_CAST = 5  # how many billed actors to use per movie

# How much each field matters. Final similarity = weighted average of the
# per-field cosine similarities. These are starting points to tune, not truths.
WEIGHTS = {
    "overview": 2.0,
    "keywords": 2.0,
    "genres": 1.5,
    "director": 1.5,
    "cast": 1.0,
}

# Small boost for well-rated movies (0.15 = at most +15% on the similarity score).
# Similarity still dominates; this only nudges ties toward better-liked movies.
QUALITY_BOOST = 0.15

COLUMNS = [
    "id", "title", "overview", "genres", "keywords", "cast", "crew",
    "release_date", "vote_average", "vote_count", "popularity",
]


# --------------------------------------------------------------------------
# 1. Loading
# --------------------------------------------------------------------------
def load_data(movies_path=MOVIES_CSV, credits_path=CREDITS_CSV):
    """Load both CSVs and join them on the movie ID (NOT on title)."""
    movies = pd.read_csv(movies_path)
    credits = pd.read_csv(credits_path)

    # credits.movie_id is the same thing as movies.id. Drop credits' title so we
    # don't get title_x / title_y columns.
    credits = credits.rename(columns={"movie_id": "id"}).drop(columns=["title"])

    # validate="one_to_one" raises an error if IDs repeat, instead of silently
    # creating duplicate rows (which merging on title can do).
    return movies.merge(credits, on="id", how="inner", validate="one_to_one")


# --------------------------------------------------------------------------
# 2. Preprocessing
# --------------------------------------------------------------------------
def parse_names(raw, limit=None):
    """Turn a JSON-like string, e.g. '[{"id": 28, "name": "Action"}]', into ['Action']."""
    if not isinstance(raw, str):  # NaN / missing -> empty list
        return []
    names = [item["name"] for item in ast.literal_eval(raw)]
    return names[:limit] if limit else names


def get_director(raw):
    """Return the director's name from the crew JSON string ('' if none)."""
    if not isinstance(raw, str):
        return ""
    for person in ast.literal_eval(raw):
        if person.get("job") == "Director":
            return person["name"]
    return ""


def normalize_title(title):
    """Lowercase, trim, and collapse repeated spaces: '  INCEPTION ' -> 'inception'."""
    return " ".join(str(title).lower().split())


def add_quality_score(df):
    """
    Add a 0-1 'quality' column using a Bayesian weighted rating (the IMDB-style formula).
    A movie with 3 votes averaging 10.0 should NOT beat one with 10,000 votes
    averaging 8.5, so ratings with few votes are pulled toward the global mean.
    """
    votes = df["vote_count"].fillna(0)
    rating = df["vote_average"].fillna(0)
    mean_rating = rating[votes > 0].mean()
    min_votes = votes.quantile(0.70)

    weighted = (votes / (votes + min_votes)) * rating + (min_votes / (votes + min_votes)) * mean_rating
    quality = (weighted - weighted.min()) / (weighted.max() - weighted.min())
    return df.assign(quality=quality)


def preprocess_data(raw):
    """Clean the merged data. Returns a DataFrame with a clean 0..N-1 index."""
    df = raw[COLUMNS].copy()  # .copy() -> no SettingWithCopyWarning later
    df = df.dropna(subset=["title"])  # a movie without a title is useless

    # Missing overview is fine: use empty text instead of throwing the movie away.
    df["overview"] = df["overview"].fillna("")

    df["genres"] = df["genres"].apply(parse_names)
    df["keywords"] = df["keywords"].apply(parse_names)
    df["cast"] = df["cast"].apply(lambda raw_cast: parse_names(raw_cast, limit=TOP_CAST))
    df["director"] = df["crew"].apply(get_director)

    df["year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year.astype("Int64")
    df["title_key"] = df["title"].apply(normalize_title)

    df = add_quality_score(df)

    # IMPORTANT: after dropping rows, labels no longer match positions.
    # reset_index makes label == position, which is what the similarity matrix uses.
    return df.drop(columns=["crew", "release_date"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# 3. Features
# --------------------------------------------------------------------------
def squash(name):
    """
    'Christopher Nolan' -> 'christophernolan', 'Science Fiction' -> 'sciencefiction'.
    Without this, 'Sam Worthington' and 'Sam Mendes' would look related because
    both contain the word 'sam'. Each name must be ONE token.
    """
    return re.sub(r"\W+", "", name.lower())


def join_tokens(names):
    return " ".join(squash(n) for n in names)


def build_features(df):
    """One text column per field, so each field can be vectorized and weighted separately."""
    return {
        "overview": df["overview"].str.lower(),
        "keywords": df["keywords"].apply(join_tokens),
        "genres": df["genres"].apply(join_tokens),
        "director": df["director"].apply(squash),
        "cast": df["cast"].apply(join_tokens),
    }


def make_vectorizer(field):
    """
    TF-IDF down-weights words that appear in many movies ('love', 'drama') and
    up-weights rare, informative ones ('heist', 'dreamwithindream').
    """
    if field == "overview":
        # Free text: drop English stop words, damp repeated words, ignore words seen once.
        return TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=2, dtype=np.float32)
    # Metadata fields are already clean tokens separated by spaces.
    return TfidfVectorizer(token_pattern=r"\S+", lowercase=False, dtype=np.float32)


def build_similarity_matrix(features, weights=WEIGHTS):
    """
    Cosine similarity per field, then a weighted average.
    Row i / column j of the result always refers to df row i / row j.
    """
    n_movies = len(next(iter(features.values())))
    similarity = np.zeros((n_movies, n_movies), dtype=np.float32)

    for field, text in features.items():
        matrix = make_vectorizer(field).fit_transform(text)  # sparse; no .toarray() needed
        similarity += weights[field] * cosine_similarity(matrix)

    similarity /= sum(weights.values())
    return similarity


# --------------------------------------------------------------------------
# 4. Searching
# --------------------------------------------------------------------------
def find_movie(df, query):
    """
    Exact, case-insensitive, whitespace-tolerant title match.
    Returns a DataFrame of matches: 0 rows = not found, 1 = unique, 2+ = ambiguous.
    The row labels are valid positions in the similarity matrix (index was reset).
    """
    return df[df["title_key"] == normalize_title(query)]


def suggest_titles(df, query, n=3):
    """Closest titles for a 'Did you mean...?' hint (never auto-selected)."""
    key = normalize_title(query)
    unique = df.drop_duplicates("title_key").set_index("title_key")["title"]
    return [unique[k] for k in get_close_matches(key, unique.index, n=n, cutoff=0.6)]


# --------------------------------------------------------------------------
# 5. Recommending
# --------------------------------------------------------------------------
def recommend(df, similarity, movie_idx, top_n=5):
    """Return the top_n most similar movies to the movie at row `movie_idx`."""
    base = similarity[movie_idx]
    # Gentle quality nudge: similarity * (1 + 0.15 * quality)
    scores = base * (1 + QUALITY_BOOST * df["quality"].to_numpy())
    scores[movie_idx] = -np.inf  # never recommend the movie itself (by position, not by "skip row 0")

    top = np.argsort(scores)[::-1][:top_n]
    result = df.iloc[top][["title", "year", "director", "genres", "vote_average"]].copy()
    result["similarity"] = base[top]
    return result


# --------------------------------------------------------------------------
# 6. Command-line interface
# --------------------------------------------------------------------------
def fmt_year(year):
    return "" if pd.isna(year) else f"({int(year)})"


def pick_match(matches):
    """If several movies share a title, show year + director and let the user choose."""
    if len(matches) == 1:
        return int(matches.index[0])

    print("Multiple movies share that title:")
    rows = list(matches.iterrows())
    for number, (_, row) in enumerate(rows, start=1):
        print(f"  {number}. {row['title']} {fmt_year(row['year'])} - dir. {row['director'] or 'unknown'}")
    while True:
        choice = input("Pick a number: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(rows):
            return int(rows[int(choice) - 1][0])
        print("Invalid choice, try again.")


def show_recommendations(df, similarity, query, top_n=5):
    matches = find_movie(df, query)
    if matches.empty:
        print("Movie not found in database.")
        hints = suggest_titles(df, query)
        if hints:
            print("Did you mean: " + ", ".join(hints) + "?")
        return

    movie_idx = pick_match(matches)
    recs = recommend(df, similarity, movie_idx, top_n)

    print(f"\nRecommended movies similar to '{df.at[movie_idx, 'title']}':\n")
    for rank, (_, row) in enumerate(recs.iterrows(), start=1):
        print(f"{rank}. {row['title']} {fmt_year(row['year'])}  [similarity {row['similarity']:.2f}]")


def load_model():
    df = preprocess_data(load_data())
    similarity = build_similarity_matrix(build_features(df))
    return df, similarity


def run_checks(df, similarity):
    """Cheap sanity checks for the bugs we fixed. Raises AssertionError on failure."""
    n = len(df)
    assert df.index.equals(pd.RangeIndex(n)), "index is not 0..N-1"
    assert similarity.shape == (n, n), "similarity matrix does not match DataFrame"
    assert np.allclose(similarity, similarity.T, atol=1e-5), "similarity matrix is not symmetric"

    # Case/space variants must all resolve to the same single movie.
    variants = ["inception", "Inception", "INCEPTION", "  Inception"]
    found = [tuple(find_movie(df, v).index) for v in variants]
    assert len(set(found)) == 1 and len(found[0]) == 1, f"variants disagree: {found}"

    assert find_movie(df, "Unknown Movie").empty, "unknown movie should not be found"

    # Each title must map to the right row, and recommendations must never include itself.
    for title in ["Inception", "12 Angry Men", "The Dark Knight", "Avatar"]:
        matches = find_movie(df, title)
        assert len(matches) == 1, f"{title}: expected 1 match, got {len(matches)}"
        idx = int(matches.index[0])
        assert df.at[idx, "title"] == title, f"{title}: wrong row {df.at[idx, 'title']}"
        recs = recommend(df, similarity, idx)
        assert title not in recs["title"].values, f"{title}: recommended itself"

    dupes = df[df.duplicated("title_key", keep=False)].sort_values("title_key")
    print(f"{n} movies loaded. Duplicate-title rows: {len(dupes)}")
    if len(dupes):
        print(dupes[["title", "year", "director"]].to_string())
    print("All checks passed.")


PANEL = ["Inception", "The Dark Knight", "Avatar", "12 Angry Men", "Toy Story", "The Godfather"]


def main():
    parser = argparse.ArgumentParser(description="TMDB movie recommender")
    parser.add_argument("movie", nargs="*", help="movie title (omit to be prompted)")
    parser.add_argument("--top", type=int, default=5, help="number of recommendations")
    parser.add_argument("--check", action="store_true", help="run sanity checks")
    parser.add_argument("--panel", action="store_true", help="show recs for a fixed list of seed movies")
    args = parser.parse_args()

    df, similarity = load_model()

    if args.check:
        run_checks(df, similarity)
    elif args.panel:
        for title in PANEL:
            show_recommendations(df, similarity, title, args.top)
            print()
    else:
        query = " ".join(args.movie) if args.movie else input("Enter movie name: ")
        show_recommendations(df, similarity, query, args.top)


if __name__ == "__main__":
    main()

