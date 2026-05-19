from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class PHashResult:
    phash: str | None
    dhash: str | None
    ahash: str | None
    available: bool
    error: str | None = None


def compute(image_path: str | Path) -> PHashResult:
    try:
        import imagehash
        from PIL import Image

        img = Image.open(image_path).convert("RGB")
        return PHashResult(
            phash=str(imagehash.phash(img)),
            dhash=str(imagehash.dhash(img)),
            ahash=str(imagehash.average_hash(img)),
            available=True,
        )
    except ImportError:
        return PHashResult(phash=None, dhash=None, ahash=None, available=False, error="imagehash not installed")
    except Exception as exc:
        return PHashResult(phash=None, dhash=None, ahash=None, available=False, error=str(exc))


def hamming_distance(hash_a: str, hash_b: str) -> int:
    """≤10 means near-duplicate for pHash/dhash."""
    try:
        import imagehash
        return imagehash.hex_to_hash(hash_a) - imagehash.hex_to_hash(hash_b)
    except Exception:
        return 64
