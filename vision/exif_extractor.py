from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Bounding boxes for Hindu pilgrimage sites mapped to their cluster.
# Format: (cluster_id, lat_min, lat_max, lon_min, lon_max, score)
_SACRED_SITES: list[tuple[str, float, float, float, float, float]] = [
    ("cluster_03", 27.50, 27.60, 77.60, 77.80, 0.9),  # Vrindavan
    ("cluster_03", 27.44, 27.52, 77.65, 77.75, 0.9),  # Mathura
    ("cluster_01", 25.28, 25.35, 82.95, 83.05, 0.9),  # Varanasi
    ("cluster_01", 30.73, 30.75, 79.06, 79.08, 0.9),  # Kedarnath
    ("cluster_06", 13.60, 13.70, 79.38, 79.45, 0.9),  # Tirupati
    ("cluster_29", 26.76, 26.80, 82.18, 82.22, 0.9),  # Ayodhya
    ("cluster_03", 19.78, 19.83, 85.81, 85.84, 0.9),  # Puri / Jagannath
    ("cluster_50", 19.76, 19.78, 74.47, 74.49, 0.9),  # Shirdi
    ("cluster_01", 29.95, 29.98, 78.14, 78.18, 0.8),  # Haridwar
    ("cluster_56", 30.09, 30.12, 78.28, 78.32, 0.8),  # Rishikesh
    ("cluster_59", 30.07, 30.10, 78.45, 78.48, 0.8),  # Devprayag
    ("cluster_51",  9.43,  9.46, 77.07, 77.10, 0.9),  # Sabarimala
    ("cluster_52", 10.47, 10.50, 78.81, 78.84, 0.9),  # Madurai / Murugan
    ("cluster_48", 18.52, 18.54, 73.84, 73.86, 0.9),  # Pune / Siddhivinayak
    ("cluster_15", 25.30, 25.32, 82.98, 83.00, 0.8),  # Sankat Mochan Varanasi
    ("cluster_59", 25.42, 25.44, 81.88, 81.90, 0.8),  # Prayagraj / Kumbh
]


@dataclass
class ExifResult:
    gps_lat: float | None = None
    gps_lon: float | None = None
    timestamp: str | None = None
    sacred_cluster: str | None = None
    sacred_score: float = 0.0


def extract(image_path: str | Path) -> ExifResult:
    try:
        from PIL import Image
        from PIL.ExifTags import TAGS, GPSTAGS

        img = Image.open(image_path)
        raw_exif = img._getexif()
        if not raw_exif:
            return ExifResult()

        exif: dict = {TAGS.get(k, k): v for k, v in raw_exif.items()}
        result = ExifResult()

        if "DateTime" in exif:
            result.timestamp = str(exif["DateTime"])

        gps_info = exif.get("GPSInfo")
        if gps_info:
            gps: dict = {GPSTAGS.get(k, k): v for k, v in gps_info.items()}
            lat = _dms_to_decimal(gps.get("GPSLatitude"), gps.get("GPSLatitudeRef"))
            lon = _dms_to_decimal(gps.get("GPSLongitude"), gps.get("GPSLongitudeRef"))
            if lat is not None and lon is not None:
                result.gps_lat = lat
                result.gps_lon = lon
                cluster, score = _lookup_sacred_site(lat, lon)
                result.sacred_cluster = cluster
                result.sacred_score = score

        return result
    except Exception:
        return ExifResult()


def _dms_to_decimal(dms, ref) -> float | None:
    if not dms or len(dms) < 3:
        return None
    try:
        degrees = float(dms[0])
        minutes = float(dms[1])
        seconds = float(dms[2])
        decimal = degrees + minutes / 60 + seconds / 3600
        if ref in ("S", "W"):
            decimal = -decimal
        return round(decimal, 6)
    except (TypeError, ValueError):
        return None


def _lookup_sacred_site(lat: float, lon: float) -> tuple[str | None, float]:
    for cluster_id, lat_min, lat_max, lon_min, lon_max, score in _SACRED_SITES:
        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
            return cluster_id, score
    return None, 0.0
