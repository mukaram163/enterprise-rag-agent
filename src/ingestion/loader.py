import io
from typing import List
from langchain_core.documents import Document
from pypdf import PdfReader
from src.connectors import GoogleDriveConnector

class GoogleDriveLoader:
    """Loader to fetch files from Google Drive and convert them into LangChain Document objects."""

    def __init__(self, connector: GoogleDriveConnector = None):
        self.connector = connector or GoogleDriveConnector()

    def load_file(self, file_id: str, file_name: str, mime_type: str) -> Document:
        """Download a single file from Google Drive and parse its text content."""
        text_content = ""

        # Handling Google Docs by exporting as plain text
        if mime_type == "application/vnd.google-apps.document":
            request = self.connector.service.files().export_media(
                fileId=file_id, mimeType="text/plain"
            )
            content = request.execute()
            text_content = content.decode("utf-8")

        # Handling PDF files
        elif mime_type == "application/pdf":
            file_data = self.connector.download_file_content(file_id)
            pdf_stream = io.BytesIO(file_data)
            reader = PdfReader(pdf_stream)
            text_content = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

        # Handling plain text files
        elif mime_type.startswith("text/"):
            file_data = self.connector.download_file_content(file_id)
            text_content = file_data.decode("utf-8")

        else:
            print(f"Skipping unsupported file type '{mime_type}' for file '{file_name}'.")
            return None

        return Document(
            page_content=text_content,
            metadata={"source": file_name, "file_id": file_id, "mime_type": mime_type}
        )

    def load_all_files(self, page_size: int = 10) -> List[Document]:
        """Fetch all supported files from Google Drive into LangChain Documents."""
        files = self.connector.list_files(page_size=page_size)
        documents = []

        for f in files:
            doc = self.load_file(f["id"], f["name"], f["mimeType"])
            if doc:
                documents.append(doc)

        return documents