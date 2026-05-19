from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from vision.clip_model import ClipModel, ClipResult
from vision.exif_extractor import ExifResult, extract as extract_exif
from vision.phash import PHashResult, compute as compute_phash
from vision.video_processor import VideoFrames, extract as extract_video
from vision.whisper_model import WhisperModel, WhisperResult

# Lightweight keyword→cluster mapping used when CLIP is disabled or for text signals.
# Intentionally a subset of ClusterDefinitions.java — same weights, cheaper execution.
_KEYWORD_CLUSTERS: dict[str, list[tuple[str, float]]] = {
    "cluster_01": [("shiva", 1.0), ("lingam", 0.9), ("mahadeva", 0.9), ("shankar", 0.9),
                   ("kedarnath", 0.8), ("shivaratri", 0.9), ("nataraja", 0.8), ("bhole", 0.7)],
    "cluster_03": [("krishna", 1.0), ("govinda", 0.9), ("radha", 0.9), ("vrindavan", 0.8),
                   ("jagannath", 0.9), ("hare krishna", 1.0), ("bal gopal", 0.8)],
    "cluster_06": [("vishnu", 1.0), ("venkateswara", 0.9), ("tirupati", 0.8), ("narayana", 0.9)],
    "cluster_15": [("hanuman", 1.0), ("bajrangbali", 0.9), ("sankat mochan", 0.8)],
    "cluster_29": [("ram", 0.8), ("sita", 0.8), ("ramayana", 0.9), ("ayodhya", 0.9)],
    "cluster_48": [("ganesh", 1.0), ("ganapati", 0.9), ("chaturthi", 0.9), ("modak", 0.7)],
    "cluster_49": [("saraswati", 1.0), ("basant panchami", 0.9)],
    "cluster_50": [("sai baba", 1.0), ("shirdi", 0.9)],
    "cluster_51": [("ayyappa", 1.0), ("sabarimala", 0.9)],
    "cluster_52": [("murugan", 1.0), ("karthikeya", 0.9), ("kavadi", 0.8)],
    "cluster_53": [("aarti", 0.9), ("puja", 0.9), ("prasad", 0.8), ("archana", 0.8)],
    "cluster_54": [("temple", 0.6), ("gopuram", 0.9), ("mandir", 0.7), ("shikhara", 0.8)],
    "cluster_55": [("diwali", 1.0), ("holi", 1.0), ("navratri", 0.9), ("dussehra", 0.9)],
    "cluster_56": [("yoga", 0.8), ("meditation", 0.8), ("rishikesh", 0.9), ("ashram", 0.7)],
    "cluster_57": [("bhajan", 1.0), ("kirtan", 1.0), ("satsang", 0.9), ("bhakti", 0.8)],
    "cluster_59": [("pilgrimage", 0.8), ("char dham", 1.0), ("kumbh mela", 1.0), ("yatra", 0.8)],
    "cluster_60": [("vedic", 0.9), ("sanskrit", 0.8), ("guru", 0.7), ("upanishad", 0.9)],
    "cluster_61": [("prasad", 0.8), ("laddoo", 0.8), ("bhog", 0.9)],
    "cluster_63": [("jyotish", 1.0), ("panchang", 1.0), ("rashi", 0.8), ("nakshatra", 0.8)],
}


@dataclass
class VisionResult:
    clusters: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    phash: str | None = None
    gps_lat: float | None = None
    gps_lon: float | None = None
    transcript: str = ""
    sources: list[str] = field(default_factory=list)


class ClusterMapper:
    def __init__(self):
        self._clip = ClipModel()
        self._whisper = WhisperModel()

    def score_image(self, image_path: str | Path) -> VisionResult:
        image_path = Path(image_path)

        phash_result: PHashResult = compute_phash(image_path)
        exif_result: ExifResult = extract_exif(image_path)
        clip_result: ClipResult = self._clip.score_image(image_path)

        clusters = _merge_scores(
            clip=clip_result.scores,
            keywords={},
            exif=exif_result,
        )
        sources = []
        if clip_result.available and clip_result.scores:
            sources.append("clip")
        if exif_result.sacred_cluster:
            sources.append("gps")

        return VisionResult(
            clusters=clusters,
            confidence=_confidence(clusters, sources),
            phash=phash_result.phash,
            gps_lat=exif_result.gps_lat,
            gps_lon=exif_result.gps_lon,
            sources=sources,
        )

    def score_video(self, video_path: str | Path, caption: str = "") -> VisionResult:
        video_path = Path(video_path)
        frames: VideoFrames = extract_video(video_path)

        clip_scores: dict[str, float] = {}
        if frames.frame_paths:
            for frame in frames.frame_paths:
                frame_result: ClipResult = self._clip.score_image(frame)
                for cluster_id, score in frame_result.scores.items():
                    clip_scores[cluster_id] = max(clip_scores.get(cluster_id, 0.0), score)

        whisper_result: WhisperResult = WhisperResult(available=False)
        if frames.audio_path:
            whisper_result = self._whisper.transcribe(frames.audio_path)

        combined_text = " ".join(filter(None, [caption, whisper_result.text]))
        keyword_scores = _keyword_score(combined_text)

        clusters = _merge_scores(clip=clip_scores, keywords=keyword_scores, exif=ExifResult())

        sources = []
        if clip_scores:
            sources.append("clip_video")
        if whisper_result.available and whisper_result.text:
            sources.append("whisper")
        if keyword_scores:
            sources.append("keywords")

        return VisionResult(
            clusters=clusters,
            confidence=_confidence(clusters, sources),
            transcript=whisper_result.text,
            sources=sources,
        )

    def score_text(self, caption: str) -> VisionResult:
        keyword_scores = _keyword_score(caption)
        return VisionResult(
            clusters=keyword_scores,
            confidence=_confidence(keyword_scores, ["keywords"] if keyword_scores else []),
            sources=["keywords"] if keyword_scores else [],
        )

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


def _merge_scores(
    clip: dict[str, float],
    keywords: dict[str, float],
    exif: ExifResult,
) -> dict[str, float]:
    all_ids = set(clip) | set(keywords)
    if exif.sacred_cluster:
        all_ids.add(exif.sacred_cluster)

    merged: dict[str, float] = {}
    for cid in all_ids:
        c = clip.get(cid, 0.0)
        k = keywords.get(cid, 0.0)
        g = exif.sacred_score if exif.sacred_cluster == cid else 0.0
        # Probabilistic OR: 1 - product(1 - signal_i)
        score = 1.0 - (1.0 - c) * (1.0 - k) * (1.0 - g)
        merged[cid] = round(min(score, 1.0), 4)

    return {k: v for k, v in sorted(merged.items(), key=lambda x: -x[1])}


def _confidence(clusters: dict[str, float], sources: list[str]) -> float:
    if not clusters:
        return 0.0
    top_score = max(clusters.values())
    source_bonus = min(len(sources) * 0.1, 0.3)
    return round(min(top_score + source_bonus, 1.0), 4)
