from __future__ import annotations

import tempfile
import urllib.request
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from vision.cluster_mapper import ClusterMapper, VisionResult

app = FastAPI(title="Aikyam Vision Service")
_mapper = ClusterMapper()


class ImageRequest(BaseModel):
    post_id: str
    image_url: str
    caption: str = ""


class VideoRequest(BaseModel):
    post_id: str
    video_url: str
    caption: str = ""


class TextRequest(BaseModel):
    post_id: str
    caption: str


class VisionResponse(BaseModel):
    post_id: str

    # Cluster signals
    clusters: dict[str, float]
    confidence: float
    sources: list[str]

    # File identity
    sha256: str | None = None
    size_bytes: int = 0
    mime_type: str = ""
    width: int | None = None
    height: int | None = None

    # Perceptual hashes
    phash: str | None = None
    dhash: str | None = None
    ahash: str | None = None

    # Camera / EXIF
    camera_make: str | None = None
    camera_model: str | None = None
    software: str | None = None
    date_time_original: str | None = None
    gps_lat: float | None = None
    gps_lon: float | None = None

    # Trust / integrity
    metadata_stripped: bool = False
    ai_generated: bool = False
    ai_signals: list[str] = []
    c2pa_present: bool = False
    jpeg_quality_estimate: int | None = None

    # Video-only
    video_format: str | None = None
    duration_seconds: float = 0.0
    video_codec: str | None = None
    resolution: str | None = None
    fps: float | None = None
    audio_codec: str | None = None
    sample_rate_hz: int | None = None
    creation_time: str | None = None
    device_make: str | None = None
    device_model: str | None = None
    transcript: str = ""


@app.get("/health")
def health():
    return {"status": "ok", "service": "aikyam-vision"}


@app.get("/ready")
def ready():
    return {"status": "ready", **_mapper.readiness()}


@app.post("/analyze/image", response_model=VisionResponse)
def analyze_image(req: ImageRequest):
    with tempfile.TemporaryDirectory() as tmp:
        path = _download(req.image_url, Path(tmp) / "image.jpg")
        result: VisionResult = _mapper.score_image(path, caption=req.caption)
    return _to_response(req.post_id, result)


@app.post("/analyze/video", response_model=VisionResponse)
def analyze_video(req: VideoRequest):
    with tempfile.TemporaryDirectory() as tmp:
        ext = Path(req.video_url.split("?")[0]).suffix or ".mp4"
        path = _download(req.video_url, Path(tmp) / f"video{ext}")
        result: VisionResult = _mapper.score_video(path, caption=req.caption)
    return _to_response(req.post_id, result)


@app.post("/analyze/text", response_model=VisionResponse)
def analyze_text(req: TextRequest):
    result: VisionResult = _mapper.score_text(req.caption)
    return _to_response(req.post_id, result)


def _download(url: str, dest: Path) -> Path:
    try:
        urllib.request.urlretrieve(url, dest)
        return dest
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Failed to download media: {exc}")


def _to_response(post_id: str, r: VisionResult) -> VisionResponse:
    return VisionResponse(
        post_id=post_id,
        clusters=r.clusters,
        confidence=r.confidence,
        sources=r.sources,
        sha256=r.sha256,
        size_bytes=r.size_bytes,
        mime_type=r.mime_type,
        width=r.width,
        height=r.height,
        phash=r.phash,
        dhash=r.dhash,
        ahash=r.ahash,
        camera_make=r.camera_make,
        camera_model=r.camera_model,
        software=r.software,
        date_time_original=r.date_time_original,
        gps_lat=r.gps_lat,
        gps_lon=r.gps_lon,
        metadata_stripped=r.metadata_stripped,
        ai_generated=r.ai_generated,
        ai_signals=r.ai_signals,
        c2pa_present=r.c2pa_present,
        jpeg_quality_estimate=r.jpeg_quality_estimate,
        video_format=r.video_format,
        duration_seconds=r.duration_seconds,
        video_codec=r.video_codec,
        resolution=r.resolution,
        fps=r.fps,
        audio_codec=r.audio_codec,
        sample_rate_hz=r.sample_rate_hz,
        creation_time=r.creation_time,
        device_make=r.device_make,
        device_model=r.device_model,
        transcript=r.transcript,
    )
