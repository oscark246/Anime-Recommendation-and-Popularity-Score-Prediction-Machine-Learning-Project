import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from itertools import chain
import ast

def preprocess_data(df):
    # Ensure 'genres' is a space-separated string, even if it's a list
    df['genres'] = df['genres'].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x))  # Convert list to string
    df['type'] = df['type'].apply(str)  # Ensure 'type' is a string
    df['source'] = df['source'].apply(str)  # Ensure 'source' is a string
    df['rating'] = df['rating'].fillna('Unknown')  # Handle missing values for 'rating'
    

    return df

# Function to build content-based recommendation model
def build_content_based_model(df):
    # Create a 'features' column by combining relevant columns (genres, type, source)
    df['features'] = df['genres'] + ' ' + df['type'] + ' ' + df['source'] + ' ' + df['score'] + ' ' + df['popularity'] + ' ' + df['ranked'] # Combine features (you can add more if needed)
    
    # Apply TF-IDF Vectorization on the 'features' column to convert text into numerical form
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['features'])  # Convert text to TF-IDF matrix
    
    # Compute cosine similarity between all anime items based on their TF-IDF vectors
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    return cosine_sim, df

# Function to recommend anime based on a given anime's index
def recommend_anime(title, cosine_sim, df, top_n=10):
   
    # Ensure title is a string and lowercase for case-insensitive comparison
    if not isinstance(title, str):
        return "Title should be a string. Please enter a valid anime title."
    
    title = title.lower()
    
    # Create a lowercase title column for comparison
    df['title_lower'] = df['name'].str.lower()

    # Find the index of the anime with the matching title
    idx = df.index[df['title_lower'] == title].tolist()

    if not idx:
        return "Anime not found in the dataset. Please check the title again."

    idx = idx[0]
    
    # Get the pairwise similarity scores for the given anime
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    
    # Sort the anime based on similarity scores (highest first)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the indices of the most similar anime
    sim_scores = sim_scores[1:top_n+1]  # Skip the first score because it's the anime itself
    anime_indices = [x[0] for x in sim_scores]
    
    
    recommended_anime = df.iloc[anime_indices].drop(columns=['features', 'title_lower', 'status', 'aired', 'broadcast'])
    

    return recommended_anime


def user_input_recommendation(df, cosine_sim):
    # Ask user for an anime title
    title = input("Enter an anime title to get recommendations: ")
    
    recommendations = recommend_anime(title, cosine_sim, df)

    if isinstance(recommendations, str):
        print(recommendations)
    else:
        print(f"Top recommendations based on '{title}'")
        print(recommendations)


def get_recommendations_from_features(user_genre, user_type, user_score, user_source, cosine_sim, df):
    # Create a user profile (you can use a weighted combination of features)
    user_profile = f"{user_genre} {user_type} {user_score} {user_source}"

    # Convert the user profile into a vector (based on your feature engineering)
    vectorizer = CountVectorizer(stop_words='english')
    features = df['features'].tolist() + [user_profile]
    feature_matrix = vectorizer.fit_transform(features)

    # Get the similarity between the user profile and all anime items
    cosine_sim_user = cosine_similarity(feature_matrix[-1], feature_matrix[:-1])

    # Get the indices of the top 10 most similar animes (excluding the input anime itself)
    sim_scores = list(enumerate(cosine_sim_user[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[:10]

    # Get the anime indices
    anime_indices = [i[0] for i in sim_scores]

    recommended_anime = df.iloc[anime_indices].drop(columns=['features', 'status', 'aired', 'broadcast'])

    return recommended_anime


def user_feature_input_recommendation(df, cosine_sim):
    try:
        
        # Display available genres
        print("Genres: 'Drama', 'Ecchi', 'Fantasy', 'Girls Love', 'Action', 'Comedy', 'Horror', 'Slice of Life', 'Supernatural', 'Sports', 'Gourmet', 'Suspense', 'Award Winning', 'Sci-Fi', 'Mystery', 'Avant Garde', 'Adventure', 'Boys Love', 'Romance'")

        # Get genres input (case-insensitive)
        genres_input = input("Enter genres (comma separated, e.g., Action, Comedy): ").strip().lower()

        # Apply literal_eval to convert each string representation of a list into an actual list
        df['genres'] = df['genres'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

        # Flatten the lists and get unique elements
        unique_genres = set(chain.from_iterable(df['genres']))
        valid_genres = list(unique_genres)
        
        # Convert valid genres to lowercase for case-insensitive comparison
        valid_genres_lower = [genre.lower() for genre in valid_genres]

        # Split the input and clean it
        user_genres = [genre.strip() for genre in genres_input.split(',')]

        # Validate user input
        invalid_genres = [genre for genre in user_genres if genre not in valid_genres_lower]
        valid_user_genres = [genre for genre in user_genres if genre in valid_genres_lower]

        # Display results
        if invalid_genres:
            print(f"Error: The following genres are invalid: {', '.join(invalid_genres)}.")
        if valid_user_genres:
            # Convert valid user genres back to the original case for display
            valid_user_genres_display = [valid_genres[valid_genres_lower.index(genre)] for genre in valid_user_genres]
            print(f"Valid genres: {', '.join(valid_user_genres_display)}.")
        else:
            print("No valid genres provided.")
            

        # Display available types and get input
        print("Type: 'TV', 'Movie', 'OVA', 'TV Special', 'ONA', 'Special'")
        type_input = input("Enter type (e.g., TV, Movie, OVA): ").strip().lower()
        valid_types = ['tv', 'movie', 'ova', 'tv special', 'ona', 'special']
        if type_input and type_input.lower() not in valid_types:
            print("Error: Invalid type input. Valid options are: TV, Movie, OVA, TV Special, ONA, Special.")
            type_input = None
        elif type_input:
            type_input = type_input.lower()

        # Get source input and validate
        print("Sources: 'Manga', 'Light novel', 'Web manga', '4-koma manga'")
        source_input = input("Enter source (e.g., Manga, Light novel): ").strip().lower()
        if source_input:
            source_input = [source.strip() for source in source_input.split(',')]
            valid_sources = set(df['source'].str.lower().unique())  # Get lowercase unique sources
            invalid_sources = [source for source in source_input if source not in valid_sources]

            if invalid_sources:
                print(f"Error: The following sources are invalid: {', '.join(invalid_sources)}.")
                source_input = None
        else:
            source_input = None

        # Validate score input
        score_input = input("Enter minimum score (0-10): ").strip()
        if score_input:
            try:
                score_input = float(score_input)
                if not (0 <= score_input <= 10):
                    raise ValueError("Score must be between 0 and 10.")
            except ValueError as e:
                print(f"Invalid score input: {e}")
                score_input = None
        else:
            score_input = None

        # Generate recommendations based on the features
        recommendations = get_recommendations_from_features(genres_input, type_input, score_input, source_input, cosine_sim, df)

        if recommendations.empty:
            print("No recommendations found based on your input.")
        else:
            print("Top recommendations based on your input:")
            print(recommendations[['name', 'genres', 'type', 'source', 'score']])

    except Exception as e:
        print(f"An error occurred while processing your request: {e}")





