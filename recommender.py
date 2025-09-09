import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
movies_path = os.path.join(BASE_DIR, "tmdb_5000_movies.csv")
credits_path = os.path.join(BASE_DIR, "tmdb_5000_credits.csv")



def load_and_prepare():
    if not os.path.exists(movies_path) or not os.path.exists(credits_path):
        raise FileNotFoundError("Make sure both CSV files are in the same folder as recommender.py")

    movies = pd.read_csv(movies_path)
    credits = pd.read_csv(credits_path)

    
    df = movies.merge(credits, left_on="id", right_on="movie_id")

   
    df["combined_features"] = (
        df["title_x"].fillna("")
        + " "
        + df["genres"].fillna("")
        + " "
        + df["keywords"].fillna("")
        + " "
        + df["tagline"].fillna("")
        + " "
        + df["cast"].fillna("")
        + " "
        + df["crew"].fillna("")
        + " "
        + df["overview"].fillna("")
    )

    return df


# -------- Step 2: Build similarity model --------
def build_model(df):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(df["combined_features"])

    # Compute cosine similarity
    sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Create index map (movie title → index)
    title_index_map = pd.Series(df.index, index=df["title_x"].str.lower()).drop_duplicates()

    return sim_matrix, title_index_map


# -------- Step 3: Recommendation function --------
def recommend(movie_name, df, sim_matrix, title_index_map, n=5):
    movie_name = movie_name.lower()

    if movie_name not in title_index_map:
        print(f"Movie '{movie_name}' not found in database.")
        return

    idx = title_index_map[movie_name]
    sim_scores = list(enumerate(sim_matrix[idx]))
    sim_scores_sorted = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    top_movies = sim_scores_sorted[1 : n + 1]  # skip itself

    print(f"\nTop {n} recommendations for '{df.iloc[idx]['title_x']}':\n")
    for i, score in top_movies:
        print(f"- {df.iloc[i]['title_x']}")


# -------- Main interactive flow --------
def main():
    print("Loading data and preparing the recommender (this runs only once)")
    df = load_and_prepare()
    sim_matrix, title_index_map = build_model(df)

    print("\n ready! Type a movie name to get recommendations. Type 'exit' to quit.\n")

    while True:
        movie = input("Enter a movie name (or 'exit'): ").strip()
        if movie.lower() in ("exit", "quit"):
            print("Goodbye")
            break
        if not movie:
            continue
        recommend(movie, df, sim_matrix, title_index_map, n=5)


if __name__ == "__main__":
    main()
