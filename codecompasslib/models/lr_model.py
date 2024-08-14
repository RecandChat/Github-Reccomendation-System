import os
import sys
import pandas as pd
from typing import Tuple, List
from pandas import DataFrame, concat
from numpy import ndarray, argsort
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from category_encoders import ordinal
from lightgbm_model import preprocess_data
from codecompasslib.API.drive_operations import download_csv_as_pd_dataframe, get_creds_drive
from codecompasslib.API.get_bulk_data import get_stared_repos, get_user_repos

# go up to root
# Construct the path to the root directory (one level up from embeddings)
root_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(root_dir)
real_project_dir = os.path.dirname(project_dir)
# Add the project directory to the Python path
sys.path.insert(0, real_project_dir)




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


def train_logistic_regression_model(df_merged: DataFrame, label_col: str) -> Tuple[LogisticRegression, ordinal.OrdinalEncoder]:
    """
    Trains a logistic regression model using the provided merged dataframe.

    This function trains a logistic regression model using the provided merged dataframe. It performs the following steps:
        1. Splits the merged dataframe into training, validation, and test sets.
        2. Encodes categorical columns using ordinal encoding.
        3. Trains the logistic regression model using the training data.
        4. Returns the trained logistic regression model and the ordinal encoder.

    Note: This function assumes that the merged dataframe has the following columns:
        - 'target': The target variable to be predicted.
        - 'id': An identifier column.
        - 'owner_user': A column representing the owner user.
        - 'embedding_0' to 'embedding_255': Numerical columns representing the embeddings.
        - 'language': A categorical column representing the language.
        - 'stars': A numerical column representing the number of stars.

    :param df_merged: DataFrame containing the training data.
    :param label_col: The name of the target variable column.
    :return: A tuple containing the trained logistic regression model and the ordinal encoder.
    """

    print("Training logistic regression model")

    X: DataFrame = df_merged.drop(columns=['target']) # drop columns not used for training
    y: DataFrame = df_merged[label_col]

    # Dataset is imbalanced -> make sure that the stratify parameter is set
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

    logistic_regression_model = LogisticRegression()
    logistic_regression_model.fit(train_x, train_y)

    return logistic_regression_model, ord_encoder


def generate_logistic_regression_recommendations(target_user: str, df_non_embedded: DataFrame,
                                                 df_embedded: DataFrame, number_of_recommendations: int = 10) -> list:
    """
    Generates recommendations using the logistic regression model.

    Args:
        target_user (str): The target user for whom recommendations are generated.
        df_non_embedded (DataFrame): The non-embedded data frame containing the features.
        df_embedded (DataFrame): The embedded data frame containing the features.
        number_of_recommendations (int, optional): The number of recommendations to generate. Defaults to 10.

    Returns:
        list: A list of recommendations, each containing the repository name, owner user, and prediction score.
    """
    # Preprocess data
    label_col: str = 'target'
    df_merged, starred_or_owned_by_user = preprocess_data(df_embedded, df_non_embedded, label_col, target_user)

    df_training_ready: DataFrame = df_merged.drop(columns=['id', 'owner_user'])

    logistic_regression_model: LogisticRegression
    ord_encoder: ordinal.OrdinalEncoder
    # Train logistic regression model
    logistic_regression_model, ord_encoder = train_logistic_regression_model(df_training_ready, label_col)

    # Make predictions for all repos
    full_dataset_x, full_dataset_y = encode_csv(df_training_ready, ord_encoder, label_col, "transform")
    all_preds = logistic_regression_model.predict_proba(full_dataset_x)[:, 1]

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
