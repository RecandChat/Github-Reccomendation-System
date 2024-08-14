from codecompasslib.API.helper_functions import OUTER_PATH
from pandas import DataFrame, read_csv
from io import BytesIO
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload, MediaDownloadProgress
from googleapiclient.discovery import build
from os.path import join
from tempfile import gettempdir


# If modifying these scopes, delete the file token.json.
SCOPES: list = ['https://www.googleapis.com/auth/drive']
DRIVE_ID: str = "0AL1DtB4TdEWdUk9PVA"
DATA_FOLDER: str = "13JitBJQLNgMvFwx4QJcvrmDwKOYAShVx"


def get_creds_drive() -> Credentials:
    """
    Get the credentials for the Google Drive API
    :return: None
    """
    creds: Credentials = Credentials.from_authorized_user_file(OUTER_PATH + "/secrets/token.json", SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow: InstalledAppFlow = InstalledAppFlow.from_client_secrets_file(OUTER_PATH + "/secrets/credentials.json",
                                                                               SCOPES)
            creds = flow.run_local_server(port=0)
        with open(OUTER_PATH + "/secrets/token.json", "w") as token:
            token.write(creds.to_json())
    return creds

    
def list_shared_drive_contents(creds: Credentials, folder_id: str, drive_id: str) -> bool:
    """
    List the contents of a folder within a Shared Drive.
    :param creds: Credentials object
    :param folder_id: The ID of the folder to list
    :param drive_id: The ID of the Shared Drive
    :return: A boolean indicating whether the operation was successful
    """
    service: build = build("drive", "v3", credentials=creds)
    try:
        response: dict = service.files().list(     # List files in the specified folder of the Shared Drive
            q=f"'{folder_id}' in parents",
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
            driveId=drive_id,
            corpora='drive',    # Ensure to set corpora to 'drive' when using driveId
            fields="nextPageToken, files(id, name)"
        ).execute()

        items: list = response.get('files', [])
        if not items:
            print("No files found in the folder.")
        else:
            print("\nFiles in the folder:")
            for item in items:
                print(u'{0} ({1})'.format(item['name'], item['id']))
        return True
    except Exception as error:
        print(f'An error occurred: {error}')
        return False
            

def download_csv_as_pd_dataframe(creds: Credentials, file_id: str) -> DataFrame:
    """
    Downloads a CSV file from Google Drive and returns it as a Pandas DataFrame.
    :param creds: A Google Drive API credentials object
    :param file_id: The ID of the file to download
    :return: A Pandas DataFrame with the contents of the CSV file
    """
    service: build = build("drive", "v3", credentials=creds)
    request: dict = service.files().get_media(fileId=file_id)
    fh: BytesIO = BytesIO()
    downloader: MediaIoBaseDownload = MediaIoBaseDownload(fh, request)
    done: bool = False

    while not done:
        status: MediaDownloadProgress
        status, done = downloader.next_chunk()
        print("\nDownload %d%%." % int(status.progress() * 100))

    fh.seek(0)  # The file's contents are now in fh, which we can use to create a Pandas DataFrame
    return read_csv(fh)


def upload_df_to_drive_as_csv(creds: Credentials, df: DataFrame, filename: str, folder_id: str) -> bool:
    """
    Uploads a Pandas DataFrame to Google Drive as a CSV file.
    :param creds: A Google Drive API credentials object
    :param df: A Pandas DataFrame to upload
    :param filename: A name for the CSV file
    :param folder_id: A folder ID in Google Drive to upload the file to
    :return: None
    """
    temp_dir: str = gettempdir()
    csv_file_path: str = join(temp_dir, filename)
    df.to_csv(csv_file_path, index=False)
    try:
        # Create Drive API client
        service: build = build("drive", "v3", credentials=creds)

        # Search for existing file with the same name in the specified folder
        response: dict = service.files().list(
            q=f"'{folder_id}' in parents and name = '{filename}'",
            spaces='drive',
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
            fields='files(id, name)'
        ).execute()

        files: response = response.get('files', [])

        # Define file metadata and media
        file_metadata: dict = {
            "name": filename,
            "parents": [folder_id]
        }

        media: MediaFileUpload = MediaFileUpload(csv_file_path, mimetype='text/csv', resumable=True)

        if files:
            # File exists, so update it
            for file in files:
                file_id: str = file.get('id')
                file = service.files().update(
                    fileId=file_id,
                    media_body=media,
                    fields='id',
                    supportsAllDrives=True,
                ).execute()
                print(f'\nFile updated. File ID: {file.get("id")}')
        else:
            # File does not exist, so create it
            file: dict = service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id',
                supportsAllDrives=True
            ).execute()
            print(f'\nFile uploaded. File ID: {file.get("id")}')
        return True

    except Exception as error:
        print(f"An error occurred: {error}")
        return False
