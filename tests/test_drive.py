import pytest
from codecompasslib.API.drive_old.drive_operations import (list_shared_drive_contents, download_csv_as_pd_dataframe,
                                                 upload_df_to_drive_as_csv)


def test_list_shared_drive_contents(creds, folder_id, drive_id) -> None:
    """
    This function tests the list_shared_drive_contents function.
    :param creds: A Google Drive API credentials object
    :param folder_id: The ID of the folder to list
    :param drive_id: The ID of the Shared Drive
    :return: None
    """
    flag: bool = list_shared_drive_contents(creds, folder_id, drive_id)
    assert flag


def test_download_csv_as_pd_dataframe(creds, file_id) -> None:
    """
    This function tests the download_csv_as_pd_dataframe function.
    :param creds: A Google Drive API credentials object
    :param file_id: The ID of the file to download
    :return: None
    """
    df = download_csv_as_pd_dataframe(creds, file_id)
    assert df is not None


def test_upload_df_to_drive_as_csv(creds, df, filename, folder_id) -> None:
    """
    This function tests the upload_df_to_drive_as_csv function.
    :param creds: A Google Drive API credentials object
    :param df: A Pandas DataFrame to upload
    :param filename: A name for the CSV file
    :param folder_id: A folder ID in Google Drive to upload the file to
    :return: None
    """
    flag: bool = upload_df_to_drive_as_csv(creds, df, filename, folder_id)
    assert flag
