"""
Repository Recommendation System

This script analyzes GitHub repositories to recommend repositories based on cosine similarity scores
calculated from their textual descriptions, names, and primary programming languages. It utilizes
TF-IDF vectorization and cosine similarity to determine the relevance of each repository to a given user preference.
"""

# Standard library imports
import os
import sys

# Third-party imports
import pandas as pd
from pandas import DataFrame
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Path setup
# Construct the path to the root directory (one level up from embeddings)
root_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(root_dir)
real_project_dir = os.path.dirname(project_dir)
# Add the project directory to the Python path
sys.path.insert(0, real_project_dir)

from codecompasslib.API.drive_old.drive_operations import download_csv_as_pd_dataframe, get_creds_drive

def load_data(full_data_folder_id: str) -> DataFrame:
    """
    Load the dataset from a specified filepath.
    
    Args:
        full_data_folder_id (str): data folder id of the dataframe on drive.

    Returns:
        pandas.DataFrame: The loaded DataFrame.
    """
    creds = get_creds_drive()
    df: DataFrame = download_csv_as_pd_dataframe(creds=creds, file_id=full_data_folder_id)
    return df

def clean_data(df: DataFrame) -> DataFrame:
    """
    Load and clean the dataset from a specified filepath.
    
    Args:
        df: loaded dataframe

    Returns:
        pandas.DataFrame: The cleaned DataFrame.
    """
    # grab the necessary columns
    df = df[['id', 'name', 'owner_user', 'description', 'url', 'language' ]]

    # Handling missing values in text columns
    df['description'] = df['description'].fillna('')
    df['name'] = df['name'].fillna('')
    df['language'] = df['language'].fillna('')

    return df



def recommend_repos(user_preference, df, top_n=10):
    """
    Recommend repositories based on user preferences.

    Args:
        user_preference (str): The user's preferred keywords or phrases.
        df (pandas.DataFrame): The DataFrame containing repository data.
        top_n (int, optional): Number of top recommendations to return. Defaults to 10.

    Returns:
        pandas.DataFrame: DataFrame containing top_n recommended repositories.
    """
    
    # Concatenating the text columns for vectorization
    text_data = df['name'] + " " + df['description'] + " " + df['language']

    # fit the TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorizer.fit_transform(text_data)
    
    # Vectorize the user preference
    user_pref_vector = tfidf_vectorizer.transform([user_preference])

    # Calculate cosine similarity with all repositories
    cosine_scores = cosine_similarity(user_pref_vector, tfidf_vectorizer.transform(text_data)).flatten()

    # Add the cosine similarity scores between repos and user preference text to the DataFrame
    df['cosine_similarity_score'] = cosine_scores
    
    # Get the indices of the repositories with the highest similarity scores
    top_indices = np.argsort(cosine_scores)[-top_n:][::-1]

    # Select the top n recommended repositories
    recommended_repos = df.iloc[top_indices].reset_index(drop=True)

    return recommended_repos[['name', 'description', 'language', 'cosine_similarity_score']]