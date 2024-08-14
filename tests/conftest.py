import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import pytest
from google.oauth2.credentials import Credentials
from codecompasslib.API.drive_old.drive_operations import get_creds_drive
from pandas import DataFrame
from unittest.mock import patch, Mock, mock_open
import json


@pytest.fixture
def sample_user() -> str:
    """
    Returns a sample user to be used in tests
    :return: A string representing a sample user
    """
    return 'Lukasaurus11'


@pytest.fixture
def non_existent_user() -> str:
    """
    Returns a non-existent user to be used in tests
    :return: A string representing a non-existent user
    """
    return 'ThisUserDoesNotExist1234567890'


@pytest.fixture
def sample_size() -> int:
    """
    Returns a sample size to be used in tests
    :return: 1
    """
    return 1


@pytest.fixture
def creds() -> Credentials:
    """
    Returns a Google Drive API credentials object
    :return: A Google Drive API credentials object
    """
    return get_creds_drive()


@pytest.fixture
def folder_id() -> str:
    """
    Returns a folder ID to be used in tests
    :return: A string representing a folder ID
    """
    return '13JitBJQLNgMvFwx4QJcvrmDwKOYAShVx'


@pytest.fixture
def drive_id() -> str:
    """
    Returns a drive ID to be used in tests
    :return: A string representing a drive ID
    """
    return '0AL1DtB4TdEWdUk9PVA'


@pytest.fixture
def file_id() -> str:
    """
    Returns a file ID to be used in tests
    :return: A string representing a file ID
    """
    return '1hAP9CD6iP4FSZP4RSRm2CYUrS2KF_Lhf'


@pytest.fixture
def df() -> DataFrame:
    """
    Returns a pandas DataFrame to be used in tests
    :return: A pandas DataFrame
    """
    return DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})


@pytest.fixture
def filename() -> str:
    """
    Returns a filename to be used in tests
    :return: A string representing a filename
    """
    return 'test.csv'

"""
This next section contains fixtures for the Chatbot
"""
@pytest.fixture
def api_response_with_useful_urls():
    """
    Provides a sample API response dictionary containing the 'usefulUrls' key.
    """
    return {
        "data": "some data",
        "usefulUrls": ["http://example.com"]
    }

@pytest.fixture
def api_response_without_useful_urls():
    """
    Provides a sample API response dictionary without the 'usefulUrls' key.
    """
    return {
        "data": "some data"
    }

@pytest.fixture
def mock_github_token():
    """
    Mocks the loading of a GitHub token.
    """
    # Adjust the path to match the full import path of the function you're mocking
    with patch('codecompasslib.chatbot.api_utilities.load_github_token', return_value='dummy_token'):
        yield

@pytest.fixture
def mock_response():
    """
    Creates a mock response to simulate `requests.post` being called.
    """
    mock = Mock()
    mock.status_code = 200
    mock.json.return_value = {"data": "some data"}
    return mock

@pytest.fixture
def mock_tool_definitions():
    """Provides a mock list of tool definitions."""
    return [
        {"name": "tool1", "description": "A test tool"},
        {"name": "tool2", "description": "Another test tool"}
    ]

@pytest.fixture
def mock_tools_json_file(tmpdir, mock_tool_definitions):
    """Creates a temporary JSON file with mock tool definitions."""
    file = tmpdir.join("tools.json")
    with open(file, 'w') as f:
        json.dump(mock_tool_definitions, f)
    return str(file)

@pytest.fixture
def mock_api_base_url():
    with patch('codecompasslib.chatbot.repo_info.load_askthecode_api_base_url', return_value="https://mockapi.askthecode.ai"):
        yield

@pytest.fixture
def mock_openai_key_file():
    mock_file = mock_open(read_data="test_openai_key")
    with patch('builtins.open', mock_file):
        yield

@pytest.fixture
def mock_github_token_file():
    mock_file = mock_open(read_data="test_github_token")
    with patch('builtins.open', mock_file):
        yield

@pytest.fixture
def mock_askthecode_api_base_url_file():
    mock_file = mock_open(read_data="https://mockapi.askthecode.ai")
    with patch('builtins.open', mock_file):
        yield

@pytest.fixture
def mock_assistant_instructions_file():
    mock_file = mock_open(read_data="test_instructions")
    with patch('builtins.open', mock_file):
        yield


