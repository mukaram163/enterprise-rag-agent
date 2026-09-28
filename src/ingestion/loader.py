import os
import io
from typing import List, Dict, Any
from pypdf import PdfReader
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

class DocumentLoader:
    def __init__(self, credentials_path: str = "credentials.json", token_path: str = "token.json"):
        self.credentials_path = credentials_path
        self.token_path = token_path
        self.drive_service = None

    def _get_drive_service(self):
        """Initializes and returns the Google Drive API service client."""
        if self.drive_service:
            return self.drive_service

        creds = None
        if os.path.exists(self.token_path):
            creds = Credentials.from_authorized_user_file(self.token_path, SCOPES)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(self.credentials_path, SCOPES)
                creds = flow.run_local_server(port=0)

            with open(self.token_path, "w") as token:
                token.write(creds.to_json())

        self.drive_service = build("drive", "v3", credentials=creds)
        return self.drive_service

    def load_local_file(self, file_path: str) -> Dict[str, Any]:
        """Loads and extracts text from a local PDF, DOCX, or TXT file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        file_name = os.path.basename(file_path)
        ext = os.path.splitext(file_name)[1].lower()
        content = ""

        if ext == ".pdf":
            reader = PdfReader(file_path)
            pages = [page.extract_text() for page in reader.pages if page.extract_text()]
            content = "\n".join(pages)
        elif ext == ".txt":
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

        return {
            "source": file_path,
            "file_name": file_name,
            "content": content,
            "metadata": {"file_type": ext, "file_size": os.path.getsize(file_path)}
        }

    def fetch_gdrive_file(self, file_id: str) -> Dict[str, Any]:
        """Downloads and extracts plain text from a Google Drive document/file."""
        service = self._get_drive_service()
        file_metadata = service.files().get(fileId=file_id, fields="id, name, mimeType").execute()
        mime_type = file_metadata.get("mimeType", "")
        file_name = file_metadata.get("name", "gdrive_file")

        # Export Google Docs to plain text, or download binary files directly
        if mime_type == "application/vnd.google-apps.document":
            request = service.files().export_media(fileId=file_id, mimeType="text/plain")
        else:
            request = service.files().get_media(fileId=file_id)

        file_stream = io.BytesIO()
        downloader = MediaIoBaseDownload(file_stream, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()

        file_stream.seek(0)
        
        if mime_type == "application/pdf":
            reader = PdfReader(file_stream)
            pages = [page.extract_text() for page in reader.pages if page.extract_text()]
            content = "\n".join(pages)
        else:
            content = file_stream.read().decode("utf-8", errors="ignore")

        return {
            "source": f"gdrive://{file_id}",
            "file_name": file_name,
            "content": content,
            "metadata": {"mime_type": mime_type, "gdrive_id": file_id}
        }