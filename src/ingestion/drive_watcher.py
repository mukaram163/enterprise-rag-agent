import json
import os
import logging
import threading
import time
from typing import Dict, Any, List, Optional, Callable
from googleapiclient.discovery import Resource

logger = logging.getLogger(__name__)

STATE_FILE = ".drive_sync_state.json"


class DriveWatcher:
    """Monitors Google Drive for file additions, modifications, and deletions."""

    def __init__(
        self,
        drive_service: Optional[Resource] = None,
        indexer: Any = None,
        on_new_file_callback: Optional[Callable] = None,
        state_file: str = STATE_FILE,
        poll_interval: int = 15
    ):
        self.service = drive_service
        self.drive_service = drive_service
        self.indexer = indexer
        self.on_new_file_callback = on_new_file_callback
        self.state_file = state_file
        self.poll_interval = poll_interval
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._processed_file_ids = set()

    def start(self):
        """Starts the DriveWatcher background polling thread."""
        if self._running:
            logger.info("DriveWatcher is already running.")
            return

        self._running = True
        self._thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._thread.start()
        logger.info("DriveWatcher background thread started.")

    def stop(self):
        """Stops the DriveWatcher background thread."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        logger.info("DriveWatcher background thread stopped.")

    def _watch_loop(self):
        """Polls Google Drive periodically for new PDF files."""
        while self._running:
            try:
                self.check_new_files()
            except Exception as e:
                logger.error(f"Error checking new files in DriveWatcher: {e}")

            for _ in range(self.poll_interval):
                if not self._running:
                    break
                time.sleep(1)

    def check_new_files(self):
        """Checks Google Drive for new PDFs and ingests them into vector DB and BM25."""
        srv = self.service or self.drive_service
        if not srv:
            logger.debug("DriveWatcher: drive_service not provided, skipping poll.")
            return

        try:
            results = srv.files().list(
                q="mimeType='application/pdf' and trashed=false",
                fields="files(id, name, modifiedTime)"
            ).execute()
        except Exception as e:
            logger.error(f"DriveWatcher API call failed: {e}")
            return

        files = results.get("files", [])
        for file in files:
            file_id = file.get("id")
            file_name = file.get("name")

            if file_id not in self._processed_file_ids:
                logger.info(f"DriveWatcher detected new PDF: {file_name} (ID: {file_id})")

                # 1. Ingest into PostgreSQL (Dense Index)
                if self.indexer and hasattr(self.indexer, "ingest_drive_file"):
                    self.indexer.ingest_drive_file(file_id=file_id, file_name=file_name)

                # 2. Trigger callback to re-fit BM25 (Sparse Index) in-memory without restart
                if self.on_new_file_callback:
                    self.on_new_file_callback()

                self._processed_file_ids.add(file_id)

    def get_start_page_token(self) -> str:
        """Fetch a fresh start page token from Google Drive API."""
        srv = self.service or self.drive_service
        response = srv.changes().getStartPageToken().execute()
        return response.get("startPageToken")

    def initialize_token(self) -> str:
        """Fetch and save the initial page token to local disk."""
        token = self.get_start_page_token()
        self.save_token(token)
        logger.info(f"Initialized DriveWatcher. Saved page token: {token}")
        return token

    def load_saved_token(self) -> Optional[str]:
        """Load the saved page token from local disk if it exists."""
        if not os.path.exists(self.state_file):
            return None
        with open(self.state_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("saved_page_token")

    def save_token(self, token: str) -> None:
        """Save updated page token to state file."""
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump({"saved_page_token": token}, f, indent=2)

    def fetch_changes(self) -> Dict[str, Any]:
        """Fetch changes from Google Drive since the saved page token."""
        srv = self.service or self.drive_service
        if not srv:
            return {"upserted": [], "deleted": [], "new_page_token": None}

        page_token = self.load_saved_token()
        if not page_token:
            page_token = self.initialize_token()

        changes_list = []
        new_page_token = page_token

        while page_token:
            response = srv.changes().list(
                pageToken=page_token,
                fields="nextPageToken, newStartPageToken, changes(fileId, removed, file(id, name, mimeType, trashed))"
            ).execute()

            changes_list.extend(response.get("changes", []))

            if "newStartPageToken" in response:
                new_page_token = response.get("newStartPageToken")

            page_token = response.get("nextPageToken")

        # Categorize changes
        upserted_files: List[Dict[str, Any]] = []
        deleted_file_ids: List[str] = []

        for change in changes_list:
            file_id = change.get("fileId")
            if change.get("removed") or (change.get("file") and change.get("file").get("trashed")):
                if file_id:
                    deleted_file_ids.append(file_id)
            else:
                file_data = change.get("file")
                if file_data and file_data.get("mimeType") == "application/pdf":
                    upserted_files.append(file_data)

        # Persist updated token for future syncs
        if new_page_token:
            self.save_token(new_page_token)

        return {
            "upserted": upserted_files,
            "deleted": deleted_file_ids,
            "new_page_token": new_page_token
        }


if __name__ == "__main__":
    from src.connectors.google_drive import GoogleDriveConnector

    connector = GoogleDriveConnector()
    watcher = DriveWatcher(drive_service=connector.service)

    print("Checking for Google Drive changes...")
    delta = watcher.fetch_changes()
    print(f"Upserted (Added/Modified) PDF Files: {len(delta['upserted'])}")
    print(f"Deleted File IDs: {len(delta['deleted'])}")
    print(f"Updated Page Token: {delta['new_page_token']}")