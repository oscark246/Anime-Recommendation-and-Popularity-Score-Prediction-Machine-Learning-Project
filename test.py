from datacleaning import *
from recommend import *

df = clean_data()

df = preprocess_data(df)

cosine_sim, df = build_content_based_model(df)


#Recommend top 10 anime based on the first anime in the dataset (index 0)
#recommended_anime = recommend_anime("Naruto", cosine_sim, df, top_n=10)
#Print recommended anime
#print("Top 10 Recommended Anime:")
#print(recommended_anime)

# Test the user input recommendation
#user_input_recommendation(df, cosine_sim)

# Test feature-based recommendation
user_feature_input_recommendation(df, cosine_sim)

