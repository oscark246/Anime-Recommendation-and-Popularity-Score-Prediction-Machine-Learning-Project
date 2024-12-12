import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MultiLabelBinarizer

# Load your dataset
df = pd.read_csv('C:/Users/User/Desktop/project/archive/toprankedanime.csv')

valid_genres = ['Drama', 'Ecchi', 'Fantasy', 'Girls Love', 'Action', 'Comedy', 'Horror', 
                'Slice of Life', 'Supernatural', 'Sports', 'Gourmet', 'Suspense', 
                'Award Winning', 'Sci-Fi', 'Mystery', 'Avant Garde', 'Adventure', 
                'Boys Love', 'Romance']
valid_types = ['TV', 'Movie', 'OVA', 'ONA', 'Special', 'TV Special']
valid_sources = ['Manga', 'Light novel', 'Web manga', '4-koma manga']

# Data Cleaning and Preparation
# Handle missing values (drop or fill them)
df.dropna(subset=['genres', 'type', 'source', 'score'], inplace=True)

# Convert 'genres' column from list of strings to actual list type (if needed)
df['genres'] = df['genres'].apply(lambda x: eval(x) if isinstance(x, str) else x)

# Feature Engineering
# One-hot encode 'type' and 'source'
type_dummies = pd.get_dummies(df['type'], prefix='type')
source_dummies = pd.get_dummies(df['source'], prefix='source')

# Multi-hot encode 'genres' using MultiLabelBinarizer
mlb = MultiLabelBinarizer()
genres_encoded = pd.DataFrame(mlb.fit_transform(df['genres']), columns=mlb.classes_)

# Combine all features
features = pd.concat([type_dummies, source_dummies, genres_encoded], axis=1)

# Target variable (score)
target = df['score']

combined = pd.concat([features, target], axis=1).dropna()
features = combined.iloc[:, :-1]
target = combined.iloc[:, -1]

print("Features shape:", features.shape)
print("Target shape:", target.shape)

# Split the Data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Model Training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")


# Make Predictions
def predict_popularity(model, mlb, features, valid_genres, valid_types, valid_sources):
    # Get user inputs
    genres, type_selected, source_selected = get_user_inputs(valid_genres, valid_types, valid_sources)
    
    # Ensure at least one valid input exists
    if not genres and not type_selected and not source_selected:
        print("No valid inputs provided. Unable to predict popularity.")
        return None

    # Prepare input features dictionary
    input_features = {
        'type': type_selected if type_selected else '',  # Default to an empty string if no valid type
        'source': source_selected if source_selected else '',  # Default to an empty string if no valid source
        'genres': genres if genres else []  # Default to an empty list if no valid genres
    }

    # Create a DataFrame for input features
    input_df = pd.DataFrame([input_features])

    # Encode input data to match the model's requirements
    input_encoded = pd.concat(
        [
            pd.get_dummies(input_df['type'], prefix='type'),
            pd.get_dummies(input_df['source'], prefix='source'),
            pd.DataFrame(mlb.transform([input_df['genres'][0]]), columns=mlb.classes_),
        ],
        axis=1,
    ).reindex(columns=features.columns, fill_value=0)

    # Predict popularity score
    prediction = model.predict(input_encoded)[0]

    # Display prediction
    print(f"Predicted Popularity Score: {prediction:.2f}")
    return prediction


def get_user_inputs(valid_genres, valid_types, valid_sources):
    
    normalized_genres = {genre.lower(): genre for genre in valid_genres}
    
    # Prompt for genres
    print(f"Available genres: {', '.join(valid_genres)}")
    genres_input = input("Enter genres (comma-separated, e.g., Action, Comedy): ").strip().lower()
    genres = []
    if genres_input:
        genres = [normalized_genres[genre.strip()] for genre in genres_input.split(',') if genre.strip() in normalized_genres]
        invalid_genres = [genre.strip() for genre in genres_input.split(',') if genre.strip() not in normalized_genres]
        if invalid_genres:
            print(f"Invalid genres ignored: {', '.join(invalid_genres)}")
    
    # Prompt for type
    print(f"Available types: {', '.join(valid_types)}")
    type_input = input("Enter type (e.g., TV, Movie): ").strip().lower()  # Lowercase the input
    type_selected = None
    for valid_type in valid_types:
        if type_input == valid_type.lower():  # Compare with lowercase valid types
            type_selected = valid_type
            break
    if not type_selected and type_input:
        print("Invalid type entered. Ignoring.")

    # Prompt for source
    print(f"Available sources: {', '.join(valid_sources)}")
    source_input = input("Enter source (e.g., Manga, Light novel): ").strip().lower()
    source_selected = source_input.capitalize() if source_input.capitalize() in valid_sources else None
    if not source_selected and source_input:
        print("Invalid source entered. Ignoring.")

    # Return validated inputs
    return genres, type_selected, source_selected


'''input_features = {
    'type': 'TV',
    'source': 'Manga',
    'genres': ['Action', 'Adventure', 'Fantasy']
}

predicted_score = predict_popularity(input_features)
print(f"Predicted Popularity Score: {predicted_score:.2f}")'''

predict_popularity(model, mlb, features, valid_genres, valid_types, valid_sources)