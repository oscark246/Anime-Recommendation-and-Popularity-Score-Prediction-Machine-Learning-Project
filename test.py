from datacleaning import *
from recommend import *

df = clean_data()

df = preprocess_data(df)

cosine_sim, df = build_content_based_model(df)

# Test the user input recommendation
user_input_recommendation(df, cosine_sim)

# Test feature-based recommendation
user_feature_input_recommendation(df, cosine_sim)

