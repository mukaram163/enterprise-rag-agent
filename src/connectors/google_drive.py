import os.path
from typing import List, Dict, Any
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

class GoogleDriveConnector:
    """Connector class to handle Google Drive OAuth authentication and file fetching."""

    def __init__(self, credentials_path: str = "credentials.json", token_path: str = "token.json"):
        self.credentials_path = credentials_path
        self.token_path = token_path
        self.service = self._authenticate()

    def _authenticate(self):
        """Private method: Handles OAuth token checks and user browser authorization."""
        creds = None

        if os.path.exists(self.token_path):
            creds = Credentials.from_authorized_user_file(self.token_path, SCOPES)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not os.path.exists(self.credentials_path):
                    raise FileNotFoundError(
                        f"Missing credentials file at '{self.credentials_path}'. "
                        "Please place your downloaded OAuth client secrets JSON file here."
                    )
                flow = InstalledAppFlow.from_client_secrets_file(self.credentials_path, SCOPES)
                creds = flow.run_local_server(port=0)

            with open(self.token_path, 'w') as token_file:
                token_file.write(creds.to_json())

        return build('drive', 'v3', credentials=creds)

    def list_files(self, page_size: int = 10) -> List[Dict[str, Any]]:
        """Fetch a list of recent files from the user's Google Drive."""
        results = self.service.files().list(
            pageSize=page_size,
            fields="nextPageToken, files(id, name, mimeType)"
        ).execute()
        return results.get('files', [])

    def download_file_content(self, file_id: str) -> bytes:
        """Download raw binary content of a specific file by its Drive ID."""
        request = self.service.files().get_media(fileId=file_id)
        return request.execute()