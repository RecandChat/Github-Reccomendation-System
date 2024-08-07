import os
import sys
import streamlit as st
import pandas as pd

# go up to root
# Construct the path to the root directory (one level up from embeddings)
root_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(root_dir)
real_project_dir = os.path.dirname(project_dir)
# Add the project directory to the Python path
sys.path.insert(0, real_project_dir)

import pandas as pd
import json
import redis
from typing import Tuple, List
from pandas import DataFrame, concat
from numpy import ndarray, argsort
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from category_encoders import ordinal

from codecompasslib.API.drive_operations import download_csv_as_pd_dataframe, get_creds_drive
from codecompasslib.API.get_bulk_data import get_stared_repos, get_user_repos
from codecompasslib.embeddings.generate_embedded_dataset import redis_to_dataframe


redis_client = redis.Redis(host='localhost', port=6379, db=0)

def load_non_embedded_data(fname: str) -> DataFrame:
    """
    Load non-embedded data from a local CSV file.
    :param file_path: Path to the non-embedded CSV file.
    :return: DataFrame containing non-embedded data.
    """
        # Construct the path to the root directory (one level up from embeddings)
    root_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(root_dir)
    real_project_dir = os.path.dirname(project_dir)
    # Add the project directory to the Python path
    sys.path.insert(0, real_project_dir)
    
    df_non_embedded = pd.read_csv(real_project_dir + '/data/' + fname)
    return df_non_embedded

def encode_csv(df: DataFrame, encoder, label_col: str, typ: str = "fit") -> Tuple[DataFrame, ndarray]:
    """
    Encode the categorical columns in a DataFrame using the specified encoder.
    :param df: The DataFrame to be encoded.
    :param encoder: The encoder object used for encoding the categorical columns.
    :param label_col: The name of the label column.
    :param typ: The type of encoding to perform. Defaults to "fit".
    :return: A tuple containing the encoded DataFrame and the label column values.
    """
    if typ == "fit":
        df = encoder.fit_transform(df)
    else:
        df = encoder.transform(df)
    y: ndarray = df[label_col].values
    del df[label_col]
    return df, y

def preprocess_data(df_embedded: DataFrame, df_non_embedded: DataFrame,
                    label_col: str, target_user: str) -> DataFrame:
    """
    Preprocesses the data by merging embedded and non-embedded datasets,
    converting the 'stars' column to integer, adding a target column,
    and dropping unnecessary columns.

    Args:
        df_embedded (DataFrame): The embedded dataset.
        df_non_embedded (DataFrame): The non-embedded dataset.
        label_col (str): The name of the target label column.
        target_user (str): The username of the target user.

    Returns:
        DataFrame: The preprocessed dataset.
        List: List of repo IDs that are either starred or owned by the target user.
    """
    # Merge the embedded and non-embedded datasets (match based on ID), grab the column you need for training 
    df_merged: DataFrame = pd.merge(df_embedded, df_non_embedded[['id', 'stars', 'language', 'owner_user']], on='id', how='left')

    # Turn stars column into integer column
    df_merged['stars'] = df_merged['stars'].astype(int)

    # Add target column: 1 if the repo is starred or owned by the user, else 0
    owned_by_target_repo_ids: List = [item['id'] for item in get_user_repos(target_user)[0]]
    starred_repo_ids: List = [item['id'] for item in get_stared_repos(target_user)[0]]
    starred_or_owned_by_user:List = starred_repo_ids + owned_by_target_repo_ids
    df_merged[label_col] = df_merged['id'].apply(lambda x: 1 if x in starred_or_owned_by_user else 0)

    return df_merged, starred_or_owned_by_user


def train_lightGBM_model(df_merged: DataFrame, label_col: str) -> Tuple[lgb.Booster, ordinal.OrdinalEncoder]:
    """
    Trains a LightGBM model using the provided merged dataframe.

    This function trains a LightGBM model using the provided merged dataframe. It performs the following steps:
        1. Sets the hyperparameters for the LightGBM model.
        2. Splits the merged dataframe into training, validation, and test sets.
        3. Encodes categorical columns using ordinal encoding.
        4. Creates LightGBM datasets for training, validation, and test data.
        5. Trains the LightGBM model using the training data.
        6. Returns the trained LightGBM model and the ordinal encoder.

    Note: This function assumes that the merged dataframe has the following columns:
        - 'target': The target variable to be predicted.
        - 'id': An identifier column.
        - 'owner_user': A column representing the owner user.
        - 'embedding_0' to 'embedding_255': Numerical columns representing the embeddings.
        - 'language': A categorical column representing the language.
        - 'stars': A numerical column representing the number of stars.

    :param df_merged: DataFrame containing the training data.
    :param label_col: The name of the target variable column.
    :return: A tuple containing the trained LightGBM model and the ordinal encoder.
    """

    # Training LightGBM model
    MAX_LEAF: int = 64
    MIN_DATA: int = 20
    NUM_OF_TREES: int = 100
    TREE_LEARNING_RATE: float = 0.15
    EARLY_STOPPING_ROUNDS: int = 20
    METRIC: str = "auc"
    SIZE: str = "sample"

    params: dict = {
        "task": "train",
        "boosting_type": "gbdt",
        "num_class": 1,
        "objective": "binary",
        "metric": METRIC,
        "num_leaves": MAX_LEAF,
        "min_data": MIN_DATA,
        "boost_from_average": True,
        "num_threads": 20,
        "feature_fraction": 0.8,
        "learning_rate": TREE_LEARNING_RATE,
    }

    print("Training LightGBM model")

    X: DataFrame = df_merged.drop(columns=['target']) # drop columns not used for training
    y: DataFrame = df_merged[label_col]

    # Dataset is imbalaned -> make sure that the stratify parameter is set
    X_combined, X_test, y_combined, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_combined, y_combined, test_size=0.1, random_state=42,
                                                      stratify=y_combined)

    # combine X_train and y_train
    train_data = concat([X_train, y_train], axis=1)
    valid_data = concat([X_val, y_val], axis=1)
    test_data = concat([X_test, y_test], axis=1)
    
    cate_cols = ['language']
    ord_encoder: ordinal.OrdinalEncoder = ordinal.OrdinalEncoder(cols=cate_cols)

    train_x, train_y = encode_csv(train_data, ord_encoder, label_col)
    valid_x, valid_y = encode_csv(valid_data, ord_encoder, label_col, "transform")
    test_x, test_y = encode_csv(test_data, ord_encoder, label_col, "transform")

    lgb_train = lgb.Dataset(train_x, train_y.reshape(-1), params=params, categorical_feature=cate_cols)
    lgb_valid = lgb.Dataset(valid_x, valid_y.reshape(-1), reference=lgb_train, categorical_feature=cate_cols)
    lgb_test = lgb.Dataset(test_x, test_y.reshape(-1), reference=lgb_train, categorical_feature=cate_cols)
    lgb_model = lgb.train(params,
                          lgb_train,
                          num_boost_round=NUM_OF_TREES,
                          valid_sets=lgb_valid,
                          categorical_feature=cate_cols,
                          callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS)])

    return lgb_model, ord_encoder


def generate_lightGBM_recommendations(target_user: str, df_non_embedded: DataFrame,
                                      df_embedded: DataFrame, number_of_recommendations: int = 10) -> list:
    """
    Generates recommendations using the LightGBM model.
    """
    # Load both DataFrames from Redis and local CSV
    df_embedded = redis_to_dataframe()
    df_non_embedded = load_non_embedded_data('data_full.csv') 

    # Preprocess data
    label_col: str = 'target'
    df_merged, starred_or_owned_by_user = preprocess_data(df_embedded, df_non_embedded, label_col, target_user)
    
    #print columns of df_merged
    print("\ndf_merged columns:", df_merged.columns)
    #print head of df_merged
    print("\ndf_merged head:", df_merged.head())
    
    df_training_ready: DataFrame = df_merged.drop(columns=['id', 'owner_user'])
    
    lgb_model: lgb.Booster
    ord_encoder: ordinal.OrdinalEncoder
    # Train LightGBM model
    lgb_model, ord_encoder = train_lightGBM_model(df_training_ready, label_col)

    # Make predictions for all repos
    full_dataset_x, full_dataset_y = encode_csv(df_training_ready, ord_encoder, label_col, "transform")
    all_preds = lgb_model.predict(full_dataset_x)

    # Get sorted predictions with the highest one first
    top_indices = argsort(all_preds)[::-1]

    # Get the top recommendations
    recommendations: list = []
    counter: int = 0
    for index in top_indices:
        if counter == number_of_recommendations:
            break
        # disregard if the repo is already starred by the user
        if df_merged.iloc[index]['id'] in starred_or_owned_by_user:
            continue
        else:
            counter += 1
            recommendations.append((df_merged.iloc[index]['id'], df_merged.iloc[index]['owner_user'], all_preds[index]))

    return recommendations

if __name__ == "__main__":
    #Want to check datatypes of columns for non embedded and embedded
    non_embedded = load_non_embedded_data("data_full.csv")
    embedded = redis_to_dataframe()
    
    print("\nNon-embedded dataset \n\nTypes: \n")
    print(non_embedded.info())
    print("\n\n Column names: \n")
    print(non_embedded.columns)
    
    
    print("\nEmbedded dataset \n\nTypes: \n")
    print(embedded.info())
    print("\n\n Column names: \n")
    print(embedded.columns)
    
    recos = generate_lightGBM_recommendations("mercyog", non_embedded, embedded, 10)
    
    

    

    





