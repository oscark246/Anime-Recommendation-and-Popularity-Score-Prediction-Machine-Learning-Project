import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def clean_data():
    # Insert address where dataset is located
    df = pd.read_csv('C:/Users/User/Desktop/project/archive/toprankedanime.csv')

    # View basic information about the dataset
    df.info()
    
    # Display the first few rows of the dataset to see the data
    print(df.head())

    # Check for missing values
    print(df.isnull().sum())

    # Drop rows with missing values in critical columns, such as 'title'
    df[['name', 'genres', 'episodes', 'status', 'aired', 'premiered', 'broadcast', 
                  'producers', 'licensors', 'studios', 'source', 'duration', 'rating', 'score', 
                  'ranked', 'popularity', 'favorites', 'type' ]].dropna()
    

    # For other columns, you could fill missing values. For instance:
    df[['name', 'genres', 'episodes', 'status', 'aired', 'premiered', 'broadcast', 
    'producers', 'licensors', 'studios', 'source', 'duration', 'rating', 'score', 
    'ranked', 'popularity', 'favorites' ]].fillna('Unknown')  # Filling missing values with 'Unknown'

    df['score'] = df['score'].apply(pd.to_numeric, errors='coerce') # Convert ratings to numeric, making non-numeric values NaN

    df['score'].fillna(df['score'].mean())  # Replace NaN ratings with the mean rating

    # Change 'aired' column to type string
    #df['aired'] = df['aired'].astype(str)
    
    # Extract the first year and store it in a new column 'start_year'
    #df['start_year'] = df['aired'].str.extract(r'(\d{4})')

    # Extract the premiiered season and store it in a new column 'season'
    #df['season'] = df['premiered'].str.extract(r'(\w+)')

    #df['month'] = df['aired'].str.extract(r'(\w{3})')

    columns_to_str = ['name', 'genres', 'status', 'aired', 'premiered', 'broadcast', 'producers', 'licensors', 
                        'studios', 'source', 'duration', 'rating', 'type', 'ranked', 'popularity', 'episodes', 'favorites', 'score']
    
    
    print(df.head())

    for col in columns_to_str:
        df[col] = df[col].astype(str)  # Convert the column to string type
        


    #for col in columns_to_str:
        #if df[col].dtype == 'object':  # Check if the column is of type string
            #df[col] = df[col].str.split(', ')  # Split by comma and space
    

    return df




