from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field
from pathlib import Path

# AI generator signatures — ordered longest first so substring matches don't double-fire.
# "stable diffusion" must precede "diffusion"; "automatic1111" before any fragment.
_AI_SIGNATURES = [
    "stable diffusion", "stablediffusion", "automatic1111",
    "midjourney", "adobe firefly", "canva ai", "dall-e", "dalle",
    "comfyui", "novelai", "imagen", "ideogram", "leonardo",
    "runwayml", "pika", "generative",
]

# Exact round dimensions common in AI-generated images
_AI_DIMENSIONS = {
    (512, 512), (512, 768), (768, 512), (768, 1024), (1024, 768),
    (1024, 1024), (1280, 720), (1280, 960), (1024, 576), (576, 1024),
}


@dataclass
class ExifToolResult:
    raw: dict = field(default_factory=dict)

    # Camera / device
    camera_make: str | None = None
    camera_model: str | None = None
    software: str | None = None
    lens: str | None = None

    # Timestamps
    date_time_original: str | None = None
    create_date: str | None = None
    modify_date: str | None = None

    # GPS
    gps_lat: float | None = None
    gps_lon: float | None = None
    gps_altitude: float | None = None

    # Flags
    metadata_stripped: bool = False
    ai_generated: bool = False
    ai_signals: list[str] = field(default_factory=list)
    c2pa_present: bool = False

    # JPEG quantization fingerprint
    jpeg_quant_tables: dict[int, list[int]] = field(default_factory=dict)
    jpeg_quality_estimate: int | None = None

    error: str | None = None


def extract(image_path: str | Path) -> ExifToolResult:
    image_path = Path(image_path)
    result = ExifToolResult()

    raw = _run_exiftool(image_path)
    result.raw = raw

    if not raw:
        result.metadata_stripped = True
    else:
        _fill_fields(result, raw)

    _extract_jpeg_quant(result, image_path)
    _detect_ai(result, image_path)

    return result


def _run_exiftool(path: Path) -> dict:
    try:
        import exiftool
        with exiftool.ExifToolHelper() as et:
            meta = et.get_metadata(str(path))
            return meta[0] if meta else {}
    except ImportError:
        return _fallback_pil_exif(path)
    except Exception:
        return _fallback_pil_exif(path)


def _fallback_pil_exif(path: Path) -> dict:
    try:
        from PIL import Image
        from PIL.ExifTags import TAGS
        img = Image.open(path)
        raw = img._getexif()
        if not raw:
            return {}
        return {TAGS.get(k, str(k)): v for k, v in raw.items()}
    except Exception:
        return {}


def _fill_fields(result: ExifToolResult, raw: dict):
    def get(*keys):
        for k in keys:
            v = raw.get(k) or raw.get(f"EXIF:{k}") or raw.get(f"XMP:{k}")
            if v:
                return str(v)
        return None

    result.camera_make = get("Make")
    result.camera_model = get("Model")
    result.software = get("Software", "CreatorTool")
    result.lens = get("LensModel", "Lens")
    result.date_time_original = get("DateTimeOriginal", "DateTime")
    result.create_date = get("CreateDate")
    result.modify_date = get("ModifyDate")

    lat = raw.get("GPS:GPSLatitude") or raw.get("Composite:GPSLatitude") or raw.get("GPSLatitude")
    lon = raw.get("GPS:GPSLongitude") or raw.get("Composite:GPSLongitude") or raw.get("GPSLongitude")
    if lat is not None and lon is not None:
        try:
            result.gps_lat = float(lat)
            result.gps_lon = float(lon)
        except (TypeError, ValueError):
            pass

    alt = raw.get("GPS:GPSAltitude") or raw.get("GPSAltitude")
    if alt is not None:
        try:
            result.gps_altitude = float(str(alt).split()[0])
        except (TypeError, ValueError):
            pass

    raw_str = " ".join(str(v) for v in raw.values()).lower()
    result.c2pa_present = "c2pa" in raw_str or "cai" in raw_str or "contentauthenticity" in raw_str

    has_camera = bool(result.camera_make or result.camera_model)
    has_timestamps = bool(result.date_time_original or result.create_date)
    has_gps = result.gps_lat is not None
    # JFIF-only JPEGs have no camera/timestamp/GPS even with a few PIL header keys.
    # True stripped = meaningful EXIF absent (camera, timestamp, GPS all missing).
    result.metadata_stripped = not has_camera and not has_timestamps and not has_gps


def _detect_ai(result: ExifToolResult, path: Path):
    signals: list[str] = []

    fields_to_check = [
        result.software or "",
        result.raw.get("Parameters", ""),
        result.raw.get("XMP:CreatorTool", ""),
        result.raw.get("XMP:Description", ""),
        result.raw.get("Comment", ""),
    ]
    combined = " ".join(str(f) for f in fields_to_check if f).lower()

    # Scan longest signatures first; once a position is consumed skip shorter overlaps
    matched_spans: list[tuple[int, int]] = []
    for sig in _AI_SIGNATURES:
        idx = combined.find(sig)
        if idx == -1:
            continue
        end = idx + len(sig)
        if any(s <= idx < e or s < end <= e for s, e in matched_spans):
            continue  # overlaps an already-matched signature
        matched_spans.append((idx, end))
        signals.append(f"metadata:{sig}")

    if result.c2pa_present:
        signals.append("c2pa_manifest")

    if not result.camera_make and not result.camera_model and not result.metadata_stripped:
        signals.append("no_camera_info")

    try:
        from PIL import Image
        with Image.open(path) as img:
            dims = (img.width, img.height)
            if dims in _AI_DIMENSIONS:
                signals.append(f"ai_dimensions:{dims[0]}x{dims[1]}")
    except Exception:
        pass

    result.ai_signals = signals
    result.ai_generated = len(signals) >= 2 or any(
        s.startswith("metadata:") or s == "c2pa_manifest" for s in signals
    )


def _extract_jpeg_quant(result: ExifToolResult, path: Path):
    if path.suffix.lower() not in (".jpg", ".jpeg"):
        return
    try:
        from PIL import Image
        with Image.open(path) as img:
            quant = getattr(img, "quantization", None)
            if quant:
                result.jpeg_quant_tables = {k: list(v) for k, v in quant.items()}
                result.jpeg_quality_estimate = _estimate_quality(quant.get(0, []))
    except Exception:
        pass


def _estimate_quality(luma_table: list[int]) -> int | None:
    if not luma_table or len(luma_table) < 64:
        return None
    avg = sum(luma_table) / len(luma_table)
    # Empirical mapping: lower avg quantization value → higher quality
    if avg <= 2:
        return 100
    elif avg <= 8:
        return 90
    elif avg <= 16:
        return 75
    elif avg <= 32:
        return 60
    elif avg <= 64:
        return 40
    else:
        return 20
