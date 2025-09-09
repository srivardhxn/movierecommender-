# movierecommender-
A content-based movie recommendation system using the TMDB 5000 Movies dataset from Kaggle. It merges movie metadata (title, genres, keywords, tagline, cast, crew, overview), applies TF-IDF vectorization, and uses cosine similarity to suggest the most similar movies through a simple Python CLI.
Load & Merge Data

Reads tmdb_5000_movies.csv and tmdb_5000_credits.csv.

Merges them into a single dataset using movie IDs.

Feature Engineering

Creates a combined_features column using title, genres, keywords, tagline, cast, crew, and overview.

Text Vectorization

Uses TF-IDF Vectorizer to convert text data into numerical vectors.

Similarity Calculation

Computes Cosine Similarity between all movies to measure how similar they are.

Recommendation

When you enter a movie name, it finds the closest matches and prints the top 5 recommendations.