"""Google Drive API sync utility.

This script uploads the local project folder to a Google Drive folder using the
Google Drive API v3. It supports service account authentication via the
GOOGLE_APPLICATION_CREDENTIALS environment variable pointing to a JSON key.

Usage (PowerShell):
  $env:GOOGLE_APPLICATION_CREDENTIALS = "C:\\path\\to\\service_account.json"
  python gdrive_sync.py --src . --dest-folder-name trading_app
  # or, if you already know a destination folder ID
  python gdrive_sync.py --src . --dest-folder-id <FOLDER_ID>

Notes:
- Share the target Drive folder with the service account email so it can write.
- For personal My Drive, create a folder and share it with the service account.
"""

from __future__ import annotations

import argparse
import os
import mimetypes
from pathlib import Path
from typing import Dict, Optional

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload


SCOPES = [
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/drive.metadata.readonly",
]


def get_drive_service() -> object:
    credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not credentials_path or not os.path.exists(credentials_path):
        raise RuntimeError(
            "GOOGLE_APPLICATION_CREDENTIALS not set or file not found. "
            "Set it to a service account JSON path."
        )
    creds = service_account.Credentials.from_service_account_file(credentials_path, scopes=SCOPES)
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def ensure_folder(service, name: str, parent_id: Optional[str] = None) -> str:
    query = f"mimeType='application/vnd.google-apps.folder' and name='{name.replace("'", "\\'")}' and trashed=false"
    if parent_id:
        query += f" and '{parent_id}' in parents"
    res = service.files().list(q=query, spaces="drive", fields="files(id, name)").execute()
    files = res.get("files", [])
    if files:
        return files[0]["id"]
    file_metadata = {
        "name": name,
        "mimeType": "application/vnd.google-apps.folder",
    }
    if parent_id:
        file_metadata["parents"] = [parent_id]
    folder = service.files().create(body=file_metadata, fields="id").execute()
    return folder["id"]


def get_children_map(service, folder_id: str) -> Dict[str, str]:
    """Return a mapping of name -> file_id for direct children of folder_id."""
    page_token = None
    name_to_id: Dict[str, str] = {}
    while True:
        res = (
            service.files()
            .list(
                q=f"'{folder_id}' in parents and trashed=false",
                spaces="drive",
                fields="nextPageToken, files(id, name, mimeType)",
                pageToken=page_token,
            )
            .execute()
        )
        for f in res.get("files", []):
            name_to_id[f["name"]] = f["id"]
        page_token = res.get("nextPageToken")
        if not page_token:
            break
    return name_to_id


def upload_file(service, local_path: Path, parent_id: str) -> str:
    mime_type, _ = mimetypes.guess_type(local_path.name)
    media = MediaFileUpload(str(local_path), mimetype=mime_type, resumable=True)
    metadata = {"name": local_path.name, "parents": [parent_id]}
    file = service.files().create(body=metadata, media_body=media, fields="id").execute()
    return file["id"]


def update_file(service, file_id: str, local_path: Path) -> None:
    mime_type, _ = mimetypes.guess_type(local_path.name)
    media = MediaFileUpload(str(local_path), mimetype=mime_type, resumable=True)
    service.files().update(fileId=file_id, media_body=media).execute()


def sync_folder(service, src: Path, dest_folder_id: str) -> None:
    """Upload all files under src to the Drive folder, creating subfolders by name.

    Strategy: for each directory level, list children in Drive and reuse by name;
    create missing folders/files; update files if sizes differ (simple heuristic).
    """
    for root, dirs, files in os.walk(src):
        rel = Path(root).relative_to(src)
        current_parent = dest_folder_id
        if rel != Path('.'):
            # create/resolve nested folders step by step
            parts = list(rel.parts)
            path_so_far = []
            for part in parts:
                path_so_far.append(part)
                # resolve under current_parent
                children = get_children_map(service, current_parent)
                if part in children:
                    current_parent = children[part]
                else:
                    current_parent = ensure_folder(service, part, parent_id=current_parent)
        # Sync files at this level
        drive_children = get_children_map(service, current_parent)
        for fname in files:
            local_file = Path(root) / fname
            # Skip typical cache/venv files
            if any(s in local_file.parts for s in [".git", "__pycache__", "node_modules", ".venv"]):
                continue
            file_id = drive_children.get(fname)
            if file_id:
                # Simple heuristic: always update (could compare size/hash with additional API calls)
                update_file(service, file_id, local_file)
            else:
                upload_file(service, local_file, current_parent)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync a local folder to Google Drive using the Drive API.")
    parser.add_argument("--src", default=".", help="Source folder to upload (default: current directory)")
    parser.add_argument("--dest-folder-id", default=None, help="Destination Google Drive folder ID")
    parser.add_argument("--dest-folder-name", default=None, help="Destination folder name (created under My Drive if ID not provided)")
    args = parser.parse_args()

    service = get_drive_service()

    if args.dest_folder_id:
        dest_id = args.dest_folder_id
    else:
        name = args.dest_folder_name or "trading_app"
        dest_id = ensure_folder(service, name)

    src_path = Path(args.src).resolve()
    sync_folder(service, src_path, dest_id)
    print(f"Synced '{src_path}' to Drive folder id: {dest_id}")


if __name__ == "__main__":
    main()



