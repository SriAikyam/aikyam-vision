from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from vision.config import VIDEO_FRAME_COUNT


@dataclass
class VideoStream:
    codec: str | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    pixel_format: str | None = None
    color_space: str | None = None
    bit_rate_kbps: int | None = None


@dataclass
class AudioStream:
    codec: str | None = None
    sample_rate_hz: int | None = None
    channels: int | None = None
    bit_rate_kbps: int | None = None


@dataclass
class VideoMetadata:
    # Container
    format_name: str | None = None
    duration_seconds: float = 0.0
    size_bytes: int = 0
    overall_bit_rate_kbps: int | None = None

    # Streams
    video: VideoStream = field(default_factory=VideoStream)
    audio: AudioStream = field(default_factory=AudioStream)

    # Timestamps
    creation_time: str | None = None      # container-level
    track_creation_time: str | None = None  # track-level (more reliable on iPhone)

    # Device / location
    device_make: str | None = None
    device_model: str | None = None
    gps_lat: float | None = None
    gps_lon: float | None = None
    gps_altitude: float | None = None

    # Extracted assets
    frame_paths: list[Path] = field(default_factory=list)
    thumbnail_path: Path | None = None   # middle frame
    audio_path: Path | None = None

    error: str | None = None


def extract(video_path: str | Path, work_dir: str | Path | None = None) -> VideoMetadata:
    video_path = Path(video_path)
    tmp = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="aikyam_vision_"))

    meta = VideoMetadata()

    probe = _ffprobe(video_path)
    if probe:
        _fill_metadata(meta, probe)
    else:
        meta.error = "ffprobe failed"
        return meta

    if meta.duration_seconds <= 0:
        meta.error = "could not determine video duration"
        return meta

    _extract_frames(meta, video_path, tmp)
    _extract_audio(meta, video_path, tmp)

    return meta


def _ffprobe(path: Path) -> dict | None:
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-print_format", "json",
                "-show_format", "-show_streams",
                str(path),
            ],
            capture_output=True, text=True, timeout=30, check=True,
        )
        return json.loads(result.stdout)
    except Exception:
        return None


def _fill_metadata(meta: VideoMetadata, probe: dict):
    fmt = probe.get("format", {})
    tags = fmt.get("tags", {})

    meta.format_name = fmt.get("format_name")
    meta.duration_seconds = float(fmt.get("duration", 0))
    meta.size_bytes = int(fmt.get("size", 0))
    bit_rate = fmt.get("bit_rate")
    if bit_rate:
        meta.overall_bit_rate_kbps = int(bit_rate) // 1000

    meta.creation_time = tags.get("creation_time")

    # GPS: QuickTime/MP4 stores location as "+lat+lon+alt/" or "±DD.DDDD±DDD.DDDD±ALT/"
    location = tags.get("location") or tags.get("com.apple.quicktime.location.ISO6709")
    if location:
        gps = _parse_iso6709(location)
        if gps:
            meta.gps_lat, meta.gps_lon, meta.gps_altitude = gps

    # Device: Apple / Android leave make/model in QuickTime tags
    meta.device_make = (
        tags.get("com.apple.quicktime.make")
        or tags.get("make")
        or tags.get("android/manufacturer")
    )
    meta.device_model = (
        tags.get("com.apple.quicktime.model")
        or tags.get("model")
        or tags.get("android/model")
    )

    for stream in probe.get("streams", []):
        codec_type = stream.get("codec_type")
        if codec_type == "video" and not meta.video.codec:
            tags_s = stream.get("tags", {})
            meta.track_creation_time = tags_s.get("creation_time") or meta.track_creation_time
            fps = _parse_fps(stream.get("r_frame_rate", "0/1"))
            meta.video = VideoStream(
                codec=stream.get("codec_name"),
                width=stream.get("width"),
                height=stream.get("height"),
                fps=fps,
                pixel_format=stream.get("pix_fmt"),
                color_space=stream.get("color_space"),
                bit_rate_kbps=int(stream.get("bit_rate", 0) or 0) // 1000 or None,
            )
        elif codec_type == "audio" and not meta.audio.codec:
            meta.audio = AudioStream(
                codec=stream.get("codec_name"),
                sample_rate_hz=int(stream.get("sample_rate", 0) or 0) or None,
                channels=stream.get("channels"),
                bit_rate_kbps=int(stream.get("bit_rate", 0) or 0) // 1000 or None,
            )


def _extract_frames(meta: VideoMetadata, video_path: Path, tmp: Path):
    duration = meta.duration_seconds
    frame_paths: list[Path] = []

    for i in range(VIDEO_FRAME_COUNT):
        t = duration * (i / max(VIDEO_FRAME_COUNT - 1, 1))
        out = tmp / f"frame_{i:02d}.jpg"
        result = subprocess.run(
            ["ffmpeg", "-ss", str(t), "-i", str(video_path),
             "-frames:v", "1", "-q:v", "2", str(out), "-y"],
            capture_output=True, timeout=30,
        )
        if result.returncode == 0 and out.exists():
            frame_paths.append(out)

    meta.frame_paths = frame_paths
    if frame_paths:
        mid = len(frame_paths) // 2
        meta.thumbnail_path = frame_paths[mid]


def _extract_audio(meta: VideoMetadata, video_path: Path, tmp: Path):
    audio_out = tmp / "audio.wav"
    result = subprocess.run(
        ["ffmpeg", "-i", str(video_path), "-vn",
         "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
         str(audio_out), "-y"],
        capture_output=True, timeout=120,
    )
    if result.returncode == 0 and audio_out.exists():
        meta.audio_path = audio_out


def _parse_fps(rate_str: str) -> float | None:
    try:
        num, den = rate_str.split("/")
        return round(int(num) / int(den), 3) if int(den) else None
    except Exception:
        return None


def _parse_iso6709(s: str) -> tuple[float, float, float | None] | None:
    """Parse ISO 6709 location string e.g. '+28.6139+077.2090+216.000/'"""
    import re
    m = re.match(r'([+-]\d+\.?\d*)([+-]\d+\.?\d*)([+-]\d+\.?\d*)?', s)
    if not m:
        return None
    try:
        lat = float(m.group(1))
        lon = float(m.group(2))
        alt = float(m.group(3)) if m.group(3) else None
        return lat, lon, alt
    except ValueError:
        return None
