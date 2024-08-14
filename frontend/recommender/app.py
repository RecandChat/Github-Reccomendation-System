import os
import sys
import streamlit as st
import redis
import json
import pandas as pd

# Function to retrieve recommendations from Redis
def retrieve_recommendations_from_redis(target_user):
    try:
        # Connect to Redis
        redis_client = redis.Redis(host='localhost', port=6379, db=0)

        # Retrieve recommendations from Redis
        recommendations = redis_client.get(target_user)

        if recommendations:
            return json.loads(recommendations.decode("utf-8"))
        else:
            return None
    except Exception as e:
        st.error(f"Could not fetch recommendations from Redis: {e}")
        return None

def main():
    # Set app title
    st.title('GitHub Repo Recommendation System')

    # Input for target user
    target_user = st.text_input("Enter the target user's username:")

    # Button to retrieve and display recommendations
    if st.button('Retrieve and Display Recommendations'):
        # Retrieve recommendations from Redis
        retrieved_recommendations = retrieve_recommendations_from_redis(target_user)

        if retrieved_recommendations:
            # Display recommendations
            st.subheader("Recommendations")
            for index, repo in enumerate(retrieved_recommendations):
                name = repo[1]  # Assuming the second element in the recommendation tuple is the repo name
                description = ""  # You may need to fetch description from Redis or another source
                link = f"https://github.com/{repo[1]}/{name}"
                
                # Display recommendation details in a card-like format with shadow
                st.markdown(f"""
                <div style="background-color: #f0f0f0; padding: 10px; border-radius: 5px; margin-bottom: 10px; color: #333; box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);">
                    <h3 style="margin-bottom: 5px; color: #000;">{name}</h3>
                    <p style="color: #000;">{description}</p>
                    <a href="{link}" target="_blank" style="color: #0366d6; text-decoration: none;">View on GitHub</a>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.error("No recommendations found for the target user.")

if __name__ == "__main__":
    main()
