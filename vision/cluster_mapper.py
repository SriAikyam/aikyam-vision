from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from vision.clip_model import ClipModel, ClipResult
from vision.exiftool_extractor import ExifToolResult, extract as extract_exif
from vision.exif_extractor import ExifResult, extract as extract_gps
from vision.file_info import FileInfo, extract as extract_file_info
from vision.phash import PHashResult, compute as compute_phash
from vision.video_processor import VideoMetadata, extract as extract_video
from vision.whisper_model import WhisperModel, WhisperResult

# Lightweight keyword→cluster mapping used when CLIP is disabled or for text signals.
_KEYWORD_CLUSTERS: dict[str, list[tuple[str, float]]] = {
    "cluster_01": [
        ("shiva", 1.0), ("lingam", 0.9), ("mahadeva", 0.9), ("mahadev", 0.9),
        ("shankar", 0.9), ("kedarnath", 0.8), ("shivaratri", 0.9), ("nataraja", 0.8),
        ("bhole", 0.7), ("kashi", 0.8), ("vishwanath", 0.8), ("varanasi", 0.8),
        ("trinetra", 0.7), ("pashupatinath", 0.8), ("somnath", 0.8),
    ],
    "cluster_03": [
        ("krishna", 1.0), ("govinda", 0.9), ("radha", 0.9), ("vrindavan", 0.8),
        ("mathura", 0.8), ("jagannath", 0.9), ("hare krishna", 1.0), ("bal gopal", 0.8),
        ("dwaraka", 0.8), ("dwarka", 0.8), ("gopal", 0.8), ("kanhaiya", 0.8),
        ("govardhan", 0.8), ("brij", 0.7),
    ],
    "cluster_06": [
        ("vishnu", 1.0), ("venkateswara", 0.9), ("tirupati", 0.8), ("narayana", 0.9),
        ("balaji", 0.9), ("padmanabha", 0.8), ("trivandrum", 0.7), ("guruvayur", 0.8),
    ],
    "cluster_15": [
        ("hanuman", 1.0), ("bajrangbali", 0.9), ("sankat mochan", 0.8),
        ("anjaneya", 0.9), ("maruti", 0.8), ("pawanputra", 0.8),
    ],
    "cluster_29": [
        ("ram", 0.8), ("rama", 0.9), ("sita", 0.8), ("ramayana", 0.9),
        ("ayodhya", 0.9), ("ramnavami", 0.9), ("jai shree ram", 1.0),
        ("laxman", 0.7), ("janki", 0.8),
    ],
    "cluster_48": [
        ("ganesh", 1.0), ("ganesha", 1.0), ("ganapati", 0.9), ("chaturthi", 0.9),
        ("modak", 0.7), ("siddhivinayak", 0.9), ("vinayaka", 0.8), ("ekdanta", 0.8),
    ],
    "cluster_49": [
        ("saraswati", 1.0), ("basant panchami", 0.9), ("saraswathi", 1.0),
        ("vagdevi", 0.8),
    ],
    "cluster_50": [
        ("sai baba", 1.0), ("shirdi", 0.9), ("sai", 0.7), ("sainath", 0.9),
    ],
    "cluster_51": [
        ("ayyappa", 1.0), ("sabarimala", 0.9), ("swamiye saranam", 1.0),
        ("makaravilakku", 0.9),
    ],
    "cluster_52": [
        ("murugan", 1.0), ("karthikeya", 0.9), ("kavadi", 0.8),
        ("skanda", 0.8), ("palani", 0.8), ("vel", 0.7),
    ],
    "cluster_53": [
        ("aarti", 0.9), ("puja", 0.9), ("prasad", 0.8), ("archana", 0.8),
        ("abhishek", 0.8), ("darshan", 0.7), ("deepam", 0.8), ("homam", 0.8),
        ("havan", 0.8), ("pooja", 0.9),
    ],
    "cluster_54": [
        ("temple", 0.6), ("gopuram", 0.9), ("mandir", 0.7), ("shikhara", 0.8),
        ("sanctum", 0.7), ("garbhagriha", 0.9), ("vimana", 0.8),
    ],
    "cluster_55": [
        ("diwali", 1.0), ("holi", 1.0), ("navratri", 0.9), ("dussehra", 0.9),
        ("dasara", 0.9), ("ganesh chaturthi", 1.0), ("onam", 0.9), ("pongal", 0.9),
        ("ugadi", 0.9), ("baisakhi", 0.9), ("chhath", 0.9), ("lohri", 0.8),
    ],
    "cluster_56": [
        ("yoga", 0.8), ("meditation", 0.8), ("rishikesh", 0.9), ("ashram", 0.7),
        ("pranayama", 0.9), ("dhyana", 0.9), ("sadhana", 0.8), ("tapas", 0.7),
    ],
    "cluster_57": [
        ("bhajan", 1.0), ("kirtan", 1.0), ("satsang", 0.9), ("bhakti", 0.8),
        ("aaradhana", 0.8), ("stotram", 0.8), ("chalisa", 0.9), ("stotra", 0.8),
    ],
    "cluster_59": [
        ("pilgrimage", 0.8), ("char dham", 1.0), ("kumbh mela", 1.0), ("yatra", 0.8),
        ("tirth", 0.8), ("tirtha", 0.8), ("parikrama", 0.9), ("pradakshina", 0.8),
    ],
    "cluster_60": [
        ("vedic", 0.9), ("sanskrit", 0.8), ("guru", 0.7), ("upanishad", 0.9),
        ("veda", 0.9), ("shastra", 0.8), ("pandit", 0.7), ("acharya", 0.7),
    ],
    "cluster_61": [
        ("prasad", 0.8), ("laddoo", 0.8), ("bhog", 0.9), ("naivedyam", 0.9),
        ("panchamrit", 0.8), ("kheer", 0.6),
    ],
    "cluster_63": [
        ("jyotish", 1.0), ("panchang", 1.0), ("rashi", 0.8), ("nakshatra", 0.8),
        ("kundali", 0.9), ("horoscope", 0.7), ("muhurat", 0.9), ("graha", 0.8),
    ],
}


@dataclass
class VisionResult:
    # Cluster scores (the primary output)
    clusters: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    sources: list[str] = field(default_factory=list)

    # Image identity
    phash: str | None = None
    dhash: str | None = None
    ahash: str | None = None
    sha256: str | None = None
    size_bytes: int = 0
    mime_type: str = ""
    width: int | None = None
    height: int | None = None

    # Rich EXIF
    camera_make: str | None = None
    camera_model: str | None = None
    software: str | None = None
    date_time_original: str | None = None
    gps_lat: float | None = None
    gps_lon: float | None = None

    # Trust / integrity flags
    metadata_stripped: bool = False
    ai_generated: bool = False
    ai_signals: list[str] = field(default_factory=list)
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
    thumbnail_path: Path | None = None

    transcript: str = ""


class ClusterMapper:
    def __init__(self):
        self._clip = ClipModel()
        self._whisper = WhisperModel()

    def score_image(self, image_path: str | Path, caption: str = "") -> VisionResult:
        image_path = Path(image_path)
        result = VisionResult()

        file_info: FileInfo = extract_file_info(image_path)
        result.sha256 = file_info.sha256
        result.size_bytes = file_info.size_bytes
        result.mime_type = file_info.mime_type
        result.width = file_info.width
        result.height = file_info.height

        phash_result: PHashResult = compute_phash(image_path)
        result.phash = phash_result.phash
        result.dhash = phash_result.dhash
        result.ahash = phash_result.ahash

        exif: ExifToolResult = extract_exif(image_path)
        result.camera_make = exif.camera_make
        result.camera_model = exif.camera_model
        result.software = exif.software
        result.date_time_original = exif.date_time_original
        result.gps_lat = exif.gps_lat
        result.gps_lon = exif.gps_lon
        result.metadata_stripped = exif.metadata_stripped
        result.ai_generated = exif.ai_generated
        result.ai_signals = exif.ai_signals
        result.c2pa_present = exif.c2pa_present
        result.jpeg_quality_estimate = exif.jpeg_quality_estimate

        gps_result: ExifResult = extract_gps(image_path)
        if result.gps_lat is None:
            result.gps_lat = gps_result.gps_lat
            result.gps_lon = gps_result.gps_lon

        clip_result: ClipResult = self._clip.score_image(image_path)

        # GPS sacred geography cluster score
        gps_scores: dict[str, float] = {}
        if gps_result.sacred_cluster:
            gps_scores[gps_result.sacred_cluster] = gps_result.sacred_score
        elif exif.gps_lat is not None:
            from vision.exif_extractor import _lookup_sacred_site
            cluster, score = _lookup_sacred_site(exif.gps_lat, exif.gps_lon)
            if cluster:
                gps_scores[cluster] = score

        keyword_scores = _keyword_score(caption)

        sources = []
        if clip_result.available and clip_result.scores:
            sources.append("clip")
        if gps_scores:
            sources.append("gps")
        if keyword_scores:
            sources.append("keywords")

        result.clusters = _merge(clip=clip_result.scores, keywords=keyword_scores, gps=gps_scores)
        result.confidence = _confidence(result.clusters, sources)
        result.sources = sources
        return result

    def score_video(self, video_path: str | Path, caption: str = "") -> VisionResult:
        video_path = Path(video_path)
        result = VisionResult()

        meta: VideoMetadata = extract_video(video_path)
        result.video_format = meta.format_name
        result.duration_seconds = meta.duration_seconds
        result.size_bytes = meta.size_bytes
        result.thumbnail_path = meta.thumbnail_path
        result.creation_time = meta.creation_time or meta.track_creation_time
        result.device_make = meta.device_make
        result.device_model = meta.device_model
        result.gps_lat = meta.gps_lat
        result.gps_lon = meta.gps_lon

        if meta.video.codec:
            result.video_codec = meta.video.codec
            result.fps = meta.video.fps
            if meta.video.width and meta.video.height:
                result.resolution = f"{meta.video.width}x{meta.video.height}"
        if meta.audio.codec:
            result.audio_codec = meta.audio.codec
            result.sample_rate_hz = meta.audio.sample_rate_hz

        clip_scores: dict[str, float] = {}
        if meta.frame_paths:
            for frame in meta.frame_paths:
                frame_result: ClipResult = self._clip.score_image(frame)
                for cid, score in frame_result.scores.items():
                    clip_scores[cid] = max(clip_scores.get(cid, 0.0), score)

        whisper_result: WhisperResult = WhisperResult(available=False)
        if meta.audio_path:
            whisper_result = self._whisper.transcribe(meta.audio_path)
        result.transcript = whisper_result.text

        combined_text = " ".join(filter(None, [caption, whisper_result.text]))
        keyword_scores = _keyword_score(combined_text)

        gps_scores: dict[str, float] = {}
        if meta.gps_lat is not None and meta.gps_lon is not None:
            from vision.exif_extractor import _lookup_sacred_site
            cluster, score = _lookup_sacred_site(meta.gps_lat, meta.gps_lon)
            if cluster:
                gps_scores[cluster] = score

        sources = []
        if clip_scores:
            sources.append("clip_video")
        if whisper_result.available and whisper_result.text:
            sources.append("whisper")
        if keyword_scores:
            sources.append("keywords")
        if gps_scores:
            sources.append("gps")

        result.clusters = _merge(clip=clip_scores, keywords=keyword_scores, gps=gps_scores)
        result.confidence = _confidence(result.clusters, sources)
        result.sources = sources
        return result

    def score_text(self, caption: str) -> VisionResult:
        keyword_scores = _keyword_score(caption)
        result = VisionResult(
            clusters=keyword_scores,
            confidence=_confidence(keyword_scores, ["keywords"] if keyword_scores else []),
            sources=["keywords"] if keyword_scores else [],
        )
        return result

    def readiness(self) -> dict:
        return {
            "clip": self._clip.readiness(),
            "whisper": self._whisper.readiness(),
        }


def _keyword_score(text: str) -> dict[str, float]:
    if not text:
        return {}
    text_lower = text.lower()
    scores: dict[str, float] = {}
    for cluster_id, entries in _KEYWORD_CLUSTERS.items():
        best = 0.0
        for kw, weight in entries:
            if re.search(r'\b' + re.escape(kw) + r'\b', text_lower):
                best = max(best, weight)
        if best > 0:
            scores[cluster_id] = round(best, 4)
    return scores


def _merge(
    clip: dict[str, float],
    keywords: dict[str, float],
    gps: dict[str, float],
) -> dict[str, float]:
    all_ids = set(clip) | set(keywords) | set(gps)
    merged: dict[str, float] = {}
    for cid in all_ids:
        c = clip.get(cid, 0.0)
        k = keywords.get(cid, 0.0)
        g = gps.get(cid, 0.0)
        score = 1.0 - (1.0 - c) * (1.0 - k) * (1.0 - g)
        merged[cid] = round(min(score, 1.0), 4)
    return {k: v for k, v in sorted(merged.items(), key=lambda x: -x[1])}


def _confidence(clusters: dict[str, float], sources: list[str]) -> float:
    if not clusters:
        return 0.0
    top_score = max(clusters.values())
    source_bonus = min(len(sources) * 0.1, 0.3)
    return round(min(top_score + source_bonus, 1.0), 4)
