import os
import sys
import redis
import json
import pandas as pd

# Navigate to root directory
root_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(root_dir)
real_project_dir = os.path.dirname(project_dir)

# Add project directory to Python path
sys.path.insert(0, real_project_dir)

# Import necessary functions from codecompasslib
from codecompasslib.models.lightgbm_model import load_data


def fill_redis_with_data():
    try:
        # Load the data
        full_data_folder_id = '1Qiy9u03hUthqaoBDr4VQqhKwtLJ2O3Yd'
        full_data_embedded_folder_id = '139wi78iRzhwGZwxmI5WALoYocR-Rk9By'
        df_non_embedded, df_embedded = load_data(full_data_folder_id, full_data_embedded_folder_id)

        print("Checkpoint 1")
        # Convert DataFrames to CSV
        df_non_embedded_csv = df_non_embedded.to_csv(index=False)
        df_embedded_csv = df_embedded.to_csv(index=False)
        
        #print first 3 rows of the csv
        print("\nFirst 3 rows of the csv")
        print(df_non_embedded_csv[:3])
        
        # Convert CSV to JSON
        print("Checkpoint 2") 
        df_non_embedded_json = json.loads(df_non_embedded_csv)
        df_embedded_json = json.loads(df_embedded_csv)
        
        #print first 3 rows of the json
        print("\nFirst 3 rows of the json")
        print(df_non_embedded_json[:3])
        
        print("Checkpoint 3")

        # Connect to Redis
        redis_client = redis.Redis(host='localhost', port=6379, db=0)

        # Store the JSON strings in Redis
        print("Storing data in Redis...")
        
        print("Not embedded df saving ...")
        redis_client.set('df_non_embedded', df_non_embedded_json)
        
        print("Embedded df saving ...")
        redis_client.set('df_embedded', df_embedded_json)

        print("Data stored in Redis successfully.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    fill_redis_with_data()
