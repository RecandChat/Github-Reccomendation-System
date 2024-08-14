import os
import sys
import streamlit as st
import pandas as pd

# Navigate to root directory
root_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(root_dir)
real_project_dir = os.path.dirname(project_dir)

# Add project directory to Python path
sys.path.insert(0, real_project_dir)

# Import necessary functions from codecompasslib
from codecompasslib.models.lightgbm_model import generate_lightGBM_recommendations,load_non_embedded_data
from codecompasslib.API.redis_operations import redis_to_dataframe

# Function to load cached data
def load_cached_data():
    # Check if data is already stored in session state
    if 'cached_data' not in st.session_state:
        with st.spinner('Fetching data from the server...'):
            # Load data and cache it
            df_non_embedded = load_non_embedded_data("data_full.csv")
            df_embedded = redis_to_dataframe()
            st.session_state.cached_data = (df_non_embedded, df_embedded)  # Cache as a tuple
    return st.session_state.cached_data


def main():
    # Load the data
    df_non_embedded, df_embedded = load_cached_data()
    
    # Set app title
    st.title('GitHub Repo Recommendation System')

    # Input for target user
    target_user = st.text_input("Enter the target user's username:")

    # Button to get recommendations
    if st.button('Get Recommendations'):
        # Check if user exists in the dataset
        if target_user not in df_non_embedded['owner_user'].values:
            st.error("User not found in the dataset. Please enter a valid username.")
        else:
            # Generate recommendations
            with st.spinner('Generating recommendations...'):
                recommendations = generate_lightGBM_recommendations(target_user, df_non_embedded, df_embedded, number_of_recommendations=10)
            
            # Display recommendations
            st.subheader("Recommendations")
            for index, repo in enumerate(recommendations):
                name = df_non_embedded[df_non_embedded['id'] == repo[0]]['name'].values[0]
                description = df_non_embedded[df_non_embedded['id'] == repo[0]]['description'].values[0]
                link = f"https://github.com/{repo[1]}/{name}"
                
                # Display recommendation details in a card-like format with shadow
                st.markdown(f"""
                <div style="background-color: #f0f0f0; padding: 10px; border-radius: 5px; margin-bottom: 10px; color: #333; box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);">
                    <h3 style="margin-bottom: 5px; color: #000;">{name}</h3>
                    <p style="color: #000;">{description}</p>
                    <a href="{link}" target="_blank" style="color: #0366d6; text-decoration: none;">View on GitHub</a>
                </div>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
