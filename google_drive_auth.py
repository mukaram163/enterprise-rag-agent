import os.path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# Define read-only scope for Stage 1 (adjust if write access is needed)
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

def get_drive_service():
    """Handles OAuth 2.0 flow and returns an authorized Google Drive API service instance."""
    creds = None
    
    # Check if access/refresh tokens already exist
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
        
    # If credentials don't exist or are invalid, authenticate user
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists('credentials.json'):
                raise FileNotFoundError(
                    "Missing 'credentials.json'. Please download your OAuth client "
                    "configuration from the Google Cloud Console."
                )
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
            
        # Save credentials for future executions
        with open('token.json', 'w') as token_file:
            token_file.write(creds.to_json())

    # Build and return the Drive API client
    return build('drive', 'v3', credentials=creds)

if __name__ == '__main__':
    try:
        service = get_drive_service()
        print("Successfully authenticated with Google Drive API!\n")
        
        # Test query: Fetch top 5 files from user's Drive
        results = service.files().list(
            pageSize=5, fields="nextPageToken, files(id, name, mimeType)"
        ).execute()
        files = results.get('files', [])

        if not files:
            print("No files found in Google Drive.")
        else:
            print("Recent Files Found:")
            for file in files:
                print(f"- {file['name']} (ID: {file['id']}) [{file['mimeType']}]")
                
    except Exception as error:
        print(f"An error occurred during authentication: {error}")