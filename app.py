from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
from recommend import preprocess_data, build_content_based_model,recommend_anime, get_recommendations_from_features

# Load the model and MultiLabelBinarizer
model = joblib.load("/home/oscark246/animerec/model.pkl")
mlb = joblib.load("/home/oscark246/animerec/mlb.pkl")

# Load the recommendation dataset
anime_df = pd.read_csv("/home/oscark246/animerec/toprankedanime.csv")  # Update with the path to your dataset
anime_df = preprocess_data(anime_df)
cosine_sim, anime_df = build_content_based_model(anime_df)

# Flask app initialization
app = Flask(__name__)

# Valid values
valid_genres = ['Drama', 'Ecchi', 'Fantasy', 'Girls Love', 'Action', 'Comedy', 'Horror',
                'Slice of Life', 'Supernatural', 'Sports', 'Gourmet', 'Suspense',
                'Award Winning', 'Sci-Fi', 'Mystery', 'Avant Garde', 'Adventure',
                'Boys Love', 'Romance']
valid_types = ['TV', 'Movie', 'OVA', 'ONA', 'Special', 'TV Special']
valid_sources = ['Manga', 'Light novel', 'Web manga', '4-koma manga']

# Landing Page
@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/autocomplete', methods=['GET'])
def autocomplete():
    query = request.args.get('query', '').lower()
    if not query:
        return jsonify([])  # Return an empty list if no query is provided

    # Filter titles based on the query
    matching_titles = anime_df[anime_df['name'].str.contains(query, case=False, na=False)]['name'].head(10).tolist()
    return jsonify(matching_titles)


# Prediction Model Page
@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'GET':
        return render_template('index.html', genres=valid_genres, types=valid_types, sources=valid_sources)
    elif request.method == 'POST':
        try:
            # Get genres input
            genres_input = request.form.getlist('genres')  # This returns a list of selected genres
            type_input = request.form.get('type', '').capitalize()
            source_input = request.form.get('source', '').capitalize()

            # Validate inputs
            if not genres_input and not type_input and not source_input:
                return render_template('result.html', message="No valid input provided. Please try again.")

            # Create input DataFrame
            input_features = {
                'type': type_input,
                'source': source_input,
                'genres': genres_input
            }
            input_df = pd.DataFrame([input_features])

            # Encode inputs
            input_encoded = pd.concat(
                [
                    pd.get_dummies(input_df['type'], prefix='type'),
                    pd.get_dummies(input_df['source'], prefix='source'),
                    pd.DataFrame(mlb.transform([input_df['genres'][0]]), columns=mlb.classes_),
                ],
                axis=1,
            ).reindex(columns=model.feature_names_in_, fill_value=0)

            # Predict popularity
            prediction = model.predict(input_encoded)[0]

            return render_template('result.html', prediction=round(prediction, 2))
        except Exception as e:
            return render_template('result.html', message=f"Error: {str(e)}")

@app.route('/recommendation', methods=['GET', 'POST'])
def recommendation():
    if request.method == 'GET':
        # Return the form for the recommendation page
        return render_template('recommendation.html', genres=valid_genres, types=valid_types, sources=valid_sources)

    elif request.method == 'POST':
        try:
            recommendation_type = request.form.get('recommendation_type')

            if recommendation_type == "title":
                anime_title = request.form.get('anime_title', '').strip()
                if not anime_title:
                    return render_template('recommendation_result.html', message="Please provide a valid anime title.")

                recommendations = recommend_anime(anime_title, cosine_sim, anime_df)
                if recommendations is None or recommendations.empty:
                    return render_template('recommendation_result.html', message="No recommendations found for the given title.")

                return render_template('recommendation_result.html', recommendations=recommendations.to_dict(orient='records'))

            elif recommendation_type == "features":
                genres_input = request.form.getlist('genres')
                type_input = request.form.get('type', '').capitalize()
                source_input = request.form.get('source', '').capitalize()
                score_input = request.form.get('score', '').strip()

                if score_input:
                    try:
                        score_input = float(score_input)
                        if not (0 <= score_input <= 10):
                            return render_template('recommendation_result.html', message="Score must be between 0 and 10.")
                    except ValueError:
                        return render_template('recommendation_result.html', message="Invalid score input. Please enter a number.")

                recommendations = get_recommendations_from_features(
                    user_genre=' '.join(genres_input),
                    user_type=type_input,
                    user_score=score_input,
                    user_source=source_input,
                    cosine_sim=cosine_sim,
                    df=anime_df
                )
                if recommendations is None or recommendations.empty:
                    return render_template('recommendation_result.html', message="No recommendations found based on the provided features.")

                return render_template('recommendation_result.html', recommendations=recommendations.to_dict(orient='records'))

        except Exception as e:
            return render_template('recommendation_result.html', message=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)