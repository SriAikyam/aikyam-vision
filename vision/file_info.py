from __future__ import annotations

import hashlib
import mimetypes
from dataclasses import dataclass
from pathlib import Path


@dataclass
class FileInfo:
    size_bytes: int
    sha256: str
    mime_type: str
    width: int | None = None
    height: int | None = None
    error: str | None = None


def extract(path: str | Path) -> FileInfo:
    path = Path(path)
    try:
        size = path.stat().st_size
        sha256 = _sha256(path)
        mime = _mime(path)
        width, height = _dimensions(path, mime)
        return FileInfo(size_bytes=size, sha256=sha256, mime_type=mime, width=width, height=height)
    except Exception as exc:
        return FileInfo(size_bytes=0, sha256="", mime_type="", error=str(exc))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _mime(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    if mime:
        return mime
    try:
        import magic
        return magic.from_file(str(path), mime=True)
    except ImportError:
        return "application/octet-stream"


def _dimensions(path: Path, mime: str) -> tuple[int | None, int | None]:
    if not mime.startswith("image/"):
        return None, None
    try:
        from PIL import Image
        with Image.open(path) as img:
            return img.width, img.height
    except Exception:
        return None, None
