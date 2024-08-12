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


# generate embedded dataset using OpenAI embeddings
def generate_openAI_embedded_csv(df, column_to_embed):
    """
    Generates embeddings for a given textual column in a DataFrame and saves the embeddings to a CSV file.

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
    # remove rows with missing values (We still have a very big dataset after removing the missing values anyway)
    df_clean = df.dropna()
    
    # turn description to lowercase and remove row if description="no description" or empty string
    df_clean = df_clean[df_clean[column_to_embed].str.lower() != 'no description']
    
    # cut text if it's size exceeds 8000 tokens
    df_clean[column_to_embed] = df_clean[column_to_embed].apply(lambda x: x[:8190]) # due to openAI API limit
    
    # grab api key from secrets
    api_key = load_openai_key()
    client = openai.Client(api_key=api_key)
    
    # extract textual column as list of strings
    textual_column = df_clean[column_to_embed].values.tolist()
    
    # extract id
    ids = df_clean['id'].values.tolist()
    owner_users = df_clean['owner_user'].values.tolist()

    # create an emptry dataframe to store the embeddings
    embedding_size = len(generate_openAI_embeddings('Test textual column', client).data[0].embedding)
    
    embeddings_columns = ['embedding_' + str(i) for i in range(embedding_size)]
    df_with_embeddings = pd.DataFrame(columns=['id', 'owner_user'] + embeddings_columns)

    batch_size = 2040 # You can adjust this value based on the API limits and your requirements

    # Iterate over every batch of textual column
    for i in range(0, len(textual_column), batch_size):
        if i % (batch_size*10) == 0:
            print(f"Processing batch starting at index: {i}")
        
        # Get the current batch of textual column
        descriptions_batch = textual_column[i:i+batch_size]
        
        # Get the embeddings for the current batch
        embeddings_response = generate_openAI_embeddings(descriptions_batch, client)
            
        # Create a DataFrame for the current batch
        batch_df = pd.DataFrame(columns=['id', 'owner_user'] + embeddings_columns)
        batch_df['id'] = ids[i:i+batch_size]
        batch_df['owner_user'] = owner_users[i:i+batch_size]
        
        # Extract the embeddings and convert them into a list of lists
        embeddings_list = [embedding.embedding for embedding in embeddings_response.data]

        # Convert the list of lists into a DataFrame
        embeddings_df = pd.DataFrame(embeddings_list, dtype='float16')

        # Assuming 'batch_df' is your original DataFrame and you want to add the embeddings to it
        # Make sure 'batch_df' has the same number of rows as 'embeddings_df'
        batch_df[embeddings_columns] = embeddings_df
        
        # Save the current batch DataFrame to a CSV file
        # Mode 'a' is for append, header=False to avoid writing headers multiple times
        batch_df.to_csv('df_embedded_1208_batch.csv', mode='a', header=not i, index=False)
        
        # Optional: Free up memory by deleting the batch DataFrame if no longer needed
        del batch_df
    
    # Load the CSV file with the embeddings
    df_with_embeddings = pd.read_csv('df_embedded_1208_batch.csv')
    return df_with_embeddings
    
def main():
    # Load the dataset
    DRIVE_ID = "0AL1DtB4TdEWdUk9PVA"
    DATA_FOLDER = "13JitBJQLNgMvFwx4QJcvrmDwKOYAShVx"

    creds = get_creds_drive()
    list_shared_drive_contents(creds=creds, folder_id=DATA_FOLDER, drive_id=DRIVE_ID)
    
    df = download_csv_as_pd_dataframe(creds,"1WSgwAhzNbSqC6e_RRBDHpgpQCnGZvVcc")
    
    columns_to_retrieve = ['id', 'name', 'owner_user', 'description', 'stars', 'language']
    
    # retrieve only the columns that are needed
    df = df[columns_to_retrieve]
    
    # Define the column to embed
    column_to_embed = 'description'
    
    # Generate the embedded dataset
    df_embedded = generate_openAI_embedded_csv(df, column_to_embed)
    
    # save the dataframe with embeddings to drive
    upload_df_to_drive_as_csv(creds, df_embedded, "df_embedded_0504.csv", DATA_FOLDER)

if __name__ == "__main__":
    main()
