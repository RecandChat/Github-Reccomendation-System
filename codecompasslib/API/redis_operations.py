#ADD ARGUMENT HERE FOR EMBEDDED / NON EMBEDDED WHEN IMPLEMENTING REDIS FOR BOTH DATASETS
import json
import sys
import os
from redis import Redis
from pandas import DataFrame, concat, read_csv
from numpy import vstack


# Redis client constants
REDIS_HOST = 'localhost'
REDIS_PORT = 6379
REDIS_DB = 0

#Initialize Redis client
redis_client = Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)

def redis_to_dataframe() -> DataFrame:
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
        key_str = key

        # Get the corresponding embedding vector
        embedded_vector = redis_client.get(key_str)
        
        if embedded_vector:
            embedding_list = json.loads(embedded_vector)  # Convert from JSON string to list
            repository_id = key_str.split(":")[1]  # Extracting the repository ID from the key
            embedded_data.append({'id': float(repository_id), 'embedding': embedding_list})

    # Create a DataFrame from the collected embedded data
    df_embed = DataFrame(embedded_data)
    df_embed['id'] = df_embed['id'].astype(float)
    
    embedding_array = vstack(df_embed['embedding'].values)
    
    df_embeddings = DataFrame(embedding_array)
    df_embeddings.columns = [f"embedding_{i}" for i in range(df_embeddings.shape[1])]
    df_embeddings = df_embeddings.astype(float)

    df_embedded = concat([df_embed[['id']], df_embeddings], axis=1)

    return df_embedded

def load_non_embedded_data(fname: str) -> DataFrame:
    """
    Load non-embedded data from a local CSV file.
    :param file_path: Path to the non-embedded CSV file.
    :return: DataFrame containing non-embedded data.
    """
    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    project_dir = os.path.dirname(root_dir)
    real_project_dir = os.path.dirname(project_dir)
    # Add the project directory to the Python path
    sys.path.insert(0, real_project_dir)
    datafolder = real_project_dir + '/data/'
    
    df_non_embedded = read_csv(datafolder + fname)
    return df_non_embedded


def save_redis_to_json(file_path='redis_data.json'):
    """
    Save all Redis data to a JSON file.
    
    Parameters:
    - file_path (str): The path to the JSON file where data will be saved.
    """
    # Get all keys
    keys = redis_client.keys('*')  # Use '*' to match all keys
    print(f"Number of keys: {len(keys)}")

    # Prepare a dictionary to hold all key-value pairs
    data_dict = {}
    
    for key in keys:
        print(f"KEY: {key}")
        print("Data type:", redis_client.type(key))
        value = redis_client.get(key)  # Adjust this function according to the Redis type, e.g., get, hgetall
        
        # Store in the dictionary with value handling
        data_dict[key] = value

    # Write to a JSON file
    with open(file_path, 'w') as json_file:
        json.dump(data_dict, json_file, indent=2, ensure_ascii=False)

    print(f"Data saved to {file_path}")
    
def load_json_to_redis(file_path='redis_data.json', host='localhost', port=6379, db=0):
    """
    Load data from a JSON file into a Redis database.

    Parameters:
    - file_path (str): The path to the JSON file to be loaded.
    - host (str): The Redis server hostname.
    - port (int): The Redis server port.
    - db (int): The Redis database number.
    """

    # Open the JSON file and load its data
    with open(file_path, 'r', encoding='utf-8') as json_file:
        data_dict = json.load(json_file)

    # Iterate over each key-value pair in the loaded data and save them in Redis
    for key, value in data_dict.items():
        if value is not None:
            redis_client.set(key, value)

    print(f"Data loaded into Redis from {file_path}")
    
def load_csv_to_redis(fname="df_embedded_combined"):
    """
    Load data from a CSV file into a Redis database.

    Parameters:
    - fname (str): The name of the CSV file to be loaded (assumes it ends with '.csv').
    """
    path = datafolder + fname + '.csv'
    print("Loading from:", path)
    
    # Read the CSV file into a pandas DataFrame
    df = read_csv(path)

    # Make sure to create the embeddings_columns dynamically based on the data
    embedding_columns = [col for col in df.columns if col.startswith("embedding_")]

    # Store each embedding in Redis
    for index, row in df.iterrows():
        redis_key = f"embedded:{row['id']}"  # Use repository ID as the Redis key
        # Convert the embedding columns to a list and store as a JSON string
        redis_client.set(redis_key, json.dumps(row[embedding_columns].tolist()))
        if index % 10000 == 0:
            print(f"Stored {index} embeddings in Redis")
        
    print(f"Data loaded into Redis from {fname}.csv")

if __name__ == "__main__":
    #save_redis_to_json('redis_embedded.json')
    #load_json_to_redis('redis_embedded.json')
    #load_csv_to_redis()
    pass