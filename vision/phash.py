from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class PHashResult:
    phash: str | None
    available: bool
    error: str | None = None


def compute(image_path: str | Path) -> PHashResult:
    try:
        import imagehash
        from PIL import Image

        img = Image.open(image_path).convert("RGB")
        h = imagehash.phash(img)
        return PHashResult(phash=str(h), available=True)
    except ImportError:
        return PHashResult(phash=None, available=False, error="imagehash not installed")
    except Exception as exc:
        return PHashResult(phash=None, available=False, error=str(exc))


def hamming_distance(hash_a: str, hash_b: str) -> int:
    """Hamming distance between two hex pHash strings. ≤10 means near-duplicate."""
    try:
        import imagehash
        return imagehash.hex_to_hash(hash_a) - imagehash.hex_to_hash(hash_b)
    except Exception:
        return 64
