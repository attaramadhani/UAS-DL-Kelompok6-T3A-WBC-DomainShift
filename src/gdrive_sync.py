"""Sinkronisasi langsung hasil eksperimen Google Colab ke folder Google Drive target."""

from __future__ import annotations

import hashlib
import io
import os
import sys
from typing import Dict, Iterable, List, Optional, Tuple


TARGET_GDRIVE_FOLDER_ID = "101PbkgjCD-y_OaZPzRcqZzdgjoCf2ld8"
DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive"]
FOLDER_MIME = "application/vnd.google-apps.folder"


def is_colab() -> bool:
    """Periksa apakah kode dijalankan di lingkungan Google Colab."""
    return "google.colab" in sys.modules


def _md5_file(path: str) -> str:
    """Hitung MD5 lokal untuk menghindari unggah ulang berkas yang identik."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


class GDriveFolderSync:
    """Pengelola unggah/unduh berkas langsung ke ID folder Google Drive via Drive v3 API."""

    def __init__(
        self,
        folder_id: str = TARGET_GDRIVE_FOLDER_ID,
        milestone_subdir: str = "M3_Main_Experiments_XAI",
    ):
        self.folder_id = folder_id
        self.milestone_subdir = milestone_subdir
        self.service = None
        self.root_folder_name: str = ""
        self.milestone_folder_id: Optional[str] = None
        self._folder_cache: Dict[Tuple[str, str], str] = {}

    @property
    def connected(self) -> bool:
        """Apakah klien Drive v3 API sudah aktif dan siap menyinkronkan berkas."""
        return self.service is not None and self.milestone_folder_id is not None

    def connect(self, require_in_colab: bool = True) -> bool:
        """Autentikasi akun Google di Colab dan verifikasi izin tulis ke folder target."""
        if not is_colab():
            print("[Info] Bukan lingkungan Google Colab; penyimpanan berjalan lokal.")
            return False

        try:
            from google.colab import auth
            import google.auth
            from googleapiclient.discovery import build

            auth.authenticate_user()
            creds, _ = google.auth.default(scopes=DRIVE_SCOPES)
            self.service = build("drive", "v3", credentials=creds, cache_discovery=False)

            meta = (
                self.service.files()
                .get(
                    fileId=self.folder_id,
                    fields="id, name, mimeType, capabilities(canAddChildren)",
                    supportsAllDrives=True,
                )
                .execute()
            )
            if meta.get("mimeType") != FOLDER_MIME:
                raise ValueError(f"ID {self.folder_id} bukan folder Google Drive.")
            can_add = meta.get("capabilities", {}).get("canAddChildren", True)
            if not can_add:
                raise PermissionError(
                    f"Akun Google saat ini tidak memiliki izin menulis ke folder '{meta.get('name')}' ({self.folder_id})."
                )

            self.root_folder_name = meta.get("name", self.folder_id)
            self.milestone_folder_id = self._ensure_child_folder(self.folder_id, self.milestone_subdir)
            print(
                f"[OK] Terhubung ke Google Drive: '{self.root_folder_name}' "
                f"(ID: {self.folder_id}) -> subfolder '{self.milestone_subdir}' "
                f"(ID: {self.milestone_folder_id})"
            )
            return True
        except Exception as exc:
            self.service = None
            self.milestone_folder_id = None
            if require_in_colab:
                raise RuntimeError(f"Gagal menghubungkan Google Drive target ({self.folder_id}): {exc}") from exc
            print(f"[Peringatan] Sinkronisasi Google Drive tidak aktif: {exc}")
            return False

    def _find_child(self, parent_id: str, name: str, mime_type: Optional[str] = None) -> Optional[dict]:
        """Cari item anak berdasarkan nama persis di dalam parent folder."""
        safe_name = name.replace("\\", "\\\\").replace("'", "\\'")
        q = f"'{parent_id}' in parents and name = '{safe_name}' and trashed = false"
        if mime_type:
            q += f" and mimeType = '{mime_type}'"
        resp = (
            self.service.files()
            .list(
                q=q,
                fields="files(id, name, mimeType, md5Checksum, size)",
                pageSize=5,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            )
            .execute()
        )
        files = resp.get("files", [])
        return files[0] if files else None

    def _ensure_child_folder(self, parent_id: str, folder_name: str) -> str:
        """Ambil atau buat sub-folder di dalam parent_id secara idempoten."""
        key = (parent_id, folder_name)
        if key in self._folder_cache:
            return self._folder_cache[key]

        existing = self._find_child(parent_id, folder_name, mime_type=FOLDER_MIME)
        if existing:
            fid = existing["id"]
        else:
            created = (
                self.service.files()
                .create(
                    body={"name": folder_name, "mimeType": FOLDER_MIME, "parents": [parent_id]},
                    fields="id",
                    supportsAllDrives=True,
                )
                .execute()
            )
            fid = created["id"]
        self._folder_cache[key] = fid
        return fid

    def _ensure_rel_dir(self, rel_dir: str) -> str:
        """Pastikan hierarki sub-folder relatif tersedia di bawah subfolder milestone."""
        assert self.milestone_folder_id is not None
        cur_id = self.milestone_folder_id
        norm_dir = rel_dir.replace("\\", "/").strip("/")
        if not norm_dir or norm_dir == ".":
            return cur_id
        for part in norm_dir.split("/"):
            if part and part != ".":
                cur_id = self._ensure_child_folder(cur_id, part)
        return cur_id

    def upload_file(self, local_path: str, rel_path: Optional[str] = None, force: bool = False) -> Optional[str]:
        """Unggah atau perbarui satu berkas lokal ke Google Drive sambil menjaga struktur path."""
        if not self.connected or not os.path.isfile(local_path):
            return None

        from googleapiclient.http import MediaFileUpload

        rel_norm = (rel_path or local_path).replace("\\", "/").lstrip("./")
        rel_dir, fname = os.path.split(rel_norm)
        parent_id = self._ensure_rel_dir(rel_dir)

        local_md5 = _md5_file(local_path)
        existing = self._find_child(parent_id, fname)
        if existing and not force and existing.get("md5Checksum") == local_md5:
            return existing["id"]

        resumable = os.path.getsize(local_path) >= 5 * 1024 * 1024
        media = MediaFileUpload(local_path, resumable=resumable)
        if existing:
            updated = (
                self.service.files()
                .update(
                    fileId=existing["id"],
                    media_body=media,
                    fields="id",
                    supportsAllDrives=True,
                )
                .execute()
            )
            return updated["id"]

        created = (
            self.service.files()
            .create(
                body={"name": fname, "parents": [parent_id]},
                media_body=media,
                fields="id",
                supportsAllDrives=True,
            )
            .execute()
        )
        return created["id"]

    def sync_paths(self, paths: Iterable[str], base_dir: str = ".") -> int:
        """Sinkronkan daftar berkas atau direktori relatif terhadap base_dir ke Drive."""
        if not self.connected:
            return 0
        count = 0
        base_abs = os.path.abspath(base_dir)
        for p in paths:
            full_p = os.path.abspath(p)
            if os.path.isfile(full_p):
                rel = os.path.relpath(full_p, base_abs)
                if self.upload_file(full_p, rel_path=rel):
                    count += 1
            elif os.path.isdir(full_p):
                for root, dirs, files in os.walk(full_p):
                    dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d != "__pycache__")
                    for fname in sorted(files):
                        if fname.startswith(".") or fname.endswith(".pyc"):
                            continue
                        fpath = os.path.join(root, fname)
                        rel = os.path.relpath(fpath, base_abs)
                        if self.upload_file(fpath, rel_path=rel):
                            count += 1
        return count

    def restore_subdir(self, remote_subdir: str, local_dir: str) -> int:
        """Unduh berkas dari sub-folder Drive (mis. 'logs' atau 'weights') untuk resume."""
        if not self.connected:
            return 0

        from googleapiclient.http import MediaIoBaseDownload

        folder_id = self._ensure_rel_dir(remote_subdir)
        os.makedirs(local_dir, exist_ok=True)
        restored = 0

        def _walk_download(fid: str, out_dir: str) -> None:
            nonlocal restored
            page_token = None
            while True:
                resp = (
                    self.service.files()
                    .list(
                        q=f"'{fid}' in parents and trashed = false",
                        fields="nextPageToken, files(id, name, mimeType, md5Checksum)",
                        pageSize=100,
                        pageToken=page_token,
                        supportsAllDrives=True,
                        includeItemsFromAllDrives=True,
                    )
                    .execute()
                )
                for item in resp.get("files", []):
                    dest = os.path.join(out_dir, item["name"])
                    if item["mimeType"] == FOLDER_MIME:
                        os.makedirs(dest, exist_ok=True)
                        _walk_download(item["id"], dest)
                        continue
                    if os.path.isfile(dest) and _md5_file(dest) == item.get("md5Checksum"):
                        continue
                    req = self.service.files().get_media(fileId=item["id"], supportsAllDrives=True)
                    with io.FileIO(dest, "wb") as fh:
                        downloader = MediaIoBaseDownload(fh, req)
                        done = False
                        while not done:
                            _, done = downloader.next_chunk()
                    restored += 1
                page_token = resp.get("nextPageToken")
                if not page_token:
                    break

        _walk_download(folder_id, local_dir)
        return restored

    def list_remote_files(self) -> List[Dict[str, object]]:
        """Daftar seluruh berkas yang sudah tersimpan di subfolder milestone pada Google Drive."""
        if not self.connected:
            return []

        records: List[Dict[str, object]] = []

        def _walk(fid: str, prefix: str) -> None:
            page_token = None
            while True:
                resp = (
                    self.service.files()
                    .list(
                        q=f"'{fid}' in parents and trashed = false",
                        fields="nextPageToken, files(id, name, mimeType, size)",
                        pageSize=200,
                        pageToken=page_token,
                        supportsAllDrives=True,
                        includeItemsFromAllDrives=True,
                    )
                    .execute()
                )
                for item in resp.get("files", []):
                    rel = f"{prefix}/{item['name']}".lstrip("/")
                    if item["mimeType"] == FOLDER_MIME:
                        _walk(item["id"], rel)
                    else:
                        size_bytes = int(item.get("size", 0) or 0)
                        records.append({
                            "drive_path": f"{self.milestone_subdir}/{rel}",
                            "size_bytes": size_bytes,
                            "size_mb": round(size_bytes / (1024 * 1024), 3),
                            "file_id": item["id"],
                        })
                page_token = resp.get("nextPageToken")
                if not page_token:
                    break

        _walk(self.milestone_folder_id, "")
        records.sort(key=lambda r: str(r["drive_path"]))
        return records
