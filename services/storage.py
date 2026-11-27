import json
import logging
import os
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from fastapi.concurrency import run_in_threadpool

logger = logging.getLogger(__name__)


class SupabaseStorageClient:
    """Lightweight Supabase Storage REST client for persistent file uploads."""

    def __init__(self) -> None:
        self.base_url = (os.getenv("SUPABASE_URL") or "").strip().rstrip("/")
        self.service_key = (os.getenv("SUPABASE_SERVICE_ROLE_KEY") or "").strip()
        self.bucket = (
            os.getenv("SUPABASE_STORAGE_BUCKET") or "graph-images"
        ).strip()

    @property
    def is_configured(self) -> bool:
        return bool(self.base_url and self.service_key and self.bucket)

    def _headers(self, content_type: Optional[str] = None) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.service_key}",
            "apikey": self.service_key,
        }
        if content_type:
            headers["Content-Type"] = content_type
        return headers

    def _upload_sync(
        self, url: str, data: bytes, headers: dict[str, str]
    ) -> tuple[int, str]:
        req = Request(url, data=data, headers=headers, method="POST")
        try:
            with urlopen(req, timeout=30) as resp:
                return resp.status, resp.read().decode("utf-8", errors="replace")
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            return exc.code, body

    async def upload_bytes(
        self,
        object_path: str,
        data: bytes,
        content_type: str = "application/octet-stream",
    ) -> Optional[str]:
        """Upload raw bytes to Supabase Storage. Returns stored path or None."""
        if not self.is_configured:
            return None

        clean_path = object_path.lstrip("/")
        encoded_path = quote(clean_path, safe="/")
        url = f"{self.base_url}/storage/v1/object/{self.bucket}/{encoded_path}"
        headers = {
            **self._headers(content_type=content_type),
            "x-upsert": "true",
        }

        try:
            status_code, body = await run_in_threadpool(
                self._upload_sync, url, data, headers
            )
            if status_code in (200, 201):
                return f"{self.bucket}/{clean_path}"
            logger.warning(
                "Supabase Storage upload failed (%s): %s",
                status_code,
                body[:300],
            )
            return None
        except (URLError, OSError) as exc:
            logger.warning("Supabase Storage upload error: %s", exc)
            return None

    async def create_signed_url(
        self, object_path: str, expires_in: int = 3600
    ) -> Optional[str]:
        """Create a temporary signed URL for a private Supabase Storage object."""
        if not self.is_configured:
            return None

        clean_path = object_path.lstrip("/")
        prefix = f"{self.bucket}/"
        if clean_path.startswith(prefix):
            clean_path = clean_path[len(prefix) :]

        encoded_path = quote(clean_path, safe="/")
        url = f"{self.base_url}/storage/v1/object/sign/{self.bucket}/{encoded_path}"
        payload = json.dumps({"expiresIn": expires_in}).encode("utf-8")
        try:
            status_code, body = await run_in_threadpool(
                self._upload_sync,
                url,
                payload,
                self._headers(content_type="application/json"),
            )
            if status_code == 200:
                signed = json.loads(body).get("signedURL")
                if signed:
                    return f"{self.base_url}/storage/v1{signed}"
            return None
        except (URLError, OSError, ValueError) as exc:
            logger.warning("Supabase signed URL error: %s", exc)
            return None


storage_client = SupabaseStorageClient()
