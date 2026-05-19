from __future__ import annotations

import os
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
    clusters: dict[str, float]
    confidence: float
    phash: str | None = None
    gps_lat: float | None = None
    gps_lon: float | None = None
    transcript: str = ""
    sources: list[str]


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
        result: VisionResult = _mapper.score_image(path)
    return _response(req.post_id, result)


@app.post("/analyze/video", response_model=VisionResponse)
def analyze_video(req: VideoRequest):
    with tempfile.TemporaryDirectory() as tmp:
        ext = Path(req.video_url).suffix or ".mp4"
        path = _download(req.video_url, Path(tmp) / f"video{ext}")
        result: VisionResult = _mapper.score_video(path, caption=req.caption)
    return _response(req.post_id, result)


@app.post("/analyze/text", response_model=VisionResponse)
def analyze_text(req: TextRequest):
    result: VisionResult = _mapper.score_text(req.caption)
    return _response(req.post_id, result)


def _download(url: str, dest: Path) -> Path:
    try:
        urllib.request.urlretrieve(url, dest)
        return dest
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Failed to download media: {exc}")


def _response(post_id: str, r: VisionResult) -> VisionResponse:
    return VisionResponse(
        post_id=post_id,
        clusters=r.clusters,
        confidence=r.confidence,
        phash=r.phash,
        gps_lat=r.gps_lat,
        gps_lon=r.gps_lon,
        transcript=r.transcript,
        sources=r.sources,
    )
