import sys
import os

# Construct the path to the root directory (one level up from embeddings)
root_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(root_dir)
real_project_dir = os.path.dirname(project_dir)

# Add the project directory to the Python path
sys.path.insert(0, real_project_dir)
from codecompasslib.API.drive_operations import get_creds_drive, list_shared_drive_contents, download_csv_as_pd_dataframe, upload_df_to_drive_as_csv
from codecompasslib.embeddings.embeddings_helper_functions import generate_openAI_embeddings
from codecompasslib.models.secrets_manager import load_openai_key
import openai
import pandas as pd
import redis
import json
import numpy as np


# Initialize Redis client
redis_client = redis.Redis(host='localhost', port=6379, db=0)

# Generate embedded dataset using OpenAI embeddings
def generate_openAI_embedded_to_redis(df, column_to_embed):
    """
    Generates embeddings for a given textual column in a DataFrame and saves the embeddings to Redis.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.
        column_to_embed (str): The name of the column to generate embeddings for.

    Returns:
        pandas.DataFrame: The DataFrame with the embeddings.

    Raises:
        None

    Example:
        df = pd.DataFrame({'id': [1, 2, 3], 'text': ['Hello', 'World', 'GitHub']})
        df_with_embeddings = generate_openAI_embedded_csv(df, 'text')
    """
    # Remove rows with missing values
    df_clean = df.dropna()

    # Turn description to lowercase and remove rows if description="no description" or empty string
    df_clean = df_clean[df_clean[column_to_embed].str.lower() != 'no description']

    # Cut text if its size exceeds 8000 tokens
    df_clean[column_to_embed] = df_clean[column_to_embed].apply(lambda x: x[:8190])  # due to OpenAI API limit

    # Grab API key from secrets
    api_key = load_openai_key()
    client = openai.Client(api_key=api_key)

    # Extract textual column as list of strings
    textual_column = df_clean[column_to_embed].values.tolist()
    
    # Extract IDs and owner_users
    ids = df_clean['id'].values.tolist()
    
    owner_users = df_clean['owner_user'].values.tolist()

    # Create an empty DataFrame to store the embeddings
    embedding_size = len(generate_openAI_embeddings('Test text for embedding', client).data[0].embedding)
    
    embeddings_columns = ['embedding_' + str(i) for i in range(embedding_size)]
    df_with_embeddings = pd.DataFrame(columns=['id', 'owner_user'] + embeddings_columns)


    batch_size = 2040  # Adjust this value based on the API limits and your requirements

    # Iterate over every batch of textual column
    for i in range(0, len(textual_column), batch_size):
        if i % (batch_size * 10) == 0:
            print(f"Processing batch starting at index: {i}")

        # Get the current batch of textual column
        descriptions_batch = textual_column[i:i + batch_size]

        # Get the embeddings for the current batch
        embeddings_response = generate_openAI_embeddings(descriptions_batch, client)
        # Create a DataFrame for the current batch
        batch_df = pd.DataFrame(columns=['id', 'owner_user'] + embeddings_columns)
        batch_df['id'] = ids[i:i + batch_size]
        batch_df['owner_user'] = owner_users[i:i + batch_size]

        # Extract the embeddings and convert them into a list of lists
        embeddings_list = [embedding.embedding for embedding in embeddings_response.data]

        # Convert the list of lists into a DataFrame
        embeddings_df = pd.DataFrame(embeddings_list, dtype='float16')

        # Assuming 'batch_df' is the original DataFrame, add the embeddings to it
        batch_df[embeddings_columns] = embeddings_df

        # Store each embedding in Redis
        for idx, row in batch_df.iterrows():
            # print(f"Storing embedding for ID: {row['id']} under the key: embedded:{row['id']}")
            redis_key = f"embedded:{row['id']}"  # Use repository ID as the Redis key
            redis_client.set(redis_key, json.dumps(row[embeddings_columns].tolist()))  # Store as JSON string

    # return df_with_embeddings # MAYBE DROP THE RETURN? JUST TO LOAD THE DATA INTO REDIS (MAYBE MAKE FUNCTION TO SAVE TO REDIS FROM DF??)


#ADD ARGUMENT HERE FOR EMBEDDED / NON EMBEDDED WHEN IMPLEMENTING REDIS FOR BOTH DATASETS
def redis_to_dataframe() -> pd.DataFrame:
    """
    Retrieves embedded datasets from Redis and converts them into a DataFrame.

    Returns:
        pd.DataFrame: A DataFrame containing 'id' and 'embedding' columns.
    """
    embedded_data = []

    # Fetch all keys matching the pattern "embedded:*"
    redis_keys = redis_client.keys('embedded:*')

    for key in redis_keys:
        # Decode the key from bytes to string
        key_str = key.decode('utf-8')

        # Get the corresponding embedding vector
        embedded_vector = redis_client.get(key_str)
        
        if embedded_vector:
            embedding_list = json.loads(embedded_vector.decode('utf-8'))  # Convert from JSON string to list
            repository_id = key_str.split(":")[1]  # Extracting the repository ID from the key
            embedded_data.append({'id': float(repository_id), 'embedding': embedding_list})

    # Create a DataFrame from the collected embedded data
    df_embed = pd.DataFrame(embedded_data)
    df_embed['id'] = df_embed['id'].astype(float)
    
    embedding_array = np.vstack(df_embed['embedding'].values)
    
    df_embeddings = pd.DataFrame(embedding_array)
    df_embeddings.columns = [f"embedding_{i}" for i in range(df_embeddings.shape[1])]
    df_embeddings = df_embeddings.astype(float)

    df_embedded = pd.concat([df_embed[['id']], df_embeddings], axis=1)

    return df_embedded


#If running main script it will start generating the embeddings from the local csv
if __name__ == "__main___":
    df = pd.read_csv(f"{real_project_dir}/data/data_full.csv")
    generate_openAI_embedded_to_redis(df, 'description')
