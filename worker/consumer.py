from __future__ import annotations

import json
import logging
import tempfile
import urllib.request
from pathlib import Path

from kafka import KafkaConsumer, KafkaProducer

from vision.cluster_mapper import ClusterMapper, VisionResult
from vision.config import (
    KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_CONSUMER_GROUP,
    POST_CREATED_TOPIC,
    VISION_SCORES_TOPIC,
)

logger = logging.getLogger("aikyam.vision.worker")


class VisionWorker:
    """
    Consumes aikyam.post.created events, runs vision analysis (CLIP / Whisper /
    EXIF / keywords), then publishes cluster scores to aikyam.vision.scores.

    SimClusters' VisionScoreConsumer picks up that topic and merges the scores
    into sim:post:{postId} + the cluster index — no direct HTTP coupling.
    """

    def __init__(self):
        self._consumer = KafkaConsumer(
            POST_CREATED_TOPIC,
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            group_id=KAFKA_CONSUMER_GROUP,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            auto_offset_reset="latest",
            enable_auto_commit=True,
        )
        self._producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            acks=1,
            retries=3,
        )
        self._mapper = ClusterMapper()

    def start(self):
        logger.info("Vision worker started, listening on %s", POST_CREATED_TOPIC)
        for message in self._consumer:
            self._handle(message.value)

    CDN_BASE = "https://cdn.shriaikyam.com/media"

    def _handle(self, payload: dict):
        post_id: str = payload.get("postId", "") or payload.get("entityId", "")

        nested   = payload.get("payload", {}) or {}
        caption: str = nested.get("postText", "") or payload.get("text", "") or ""
        entity_id: str | None = payload.get("templeId") or payload.get("entityId")

        # Resolve media URLs from assetId using CDN pattern
        # assetId lives at payload.content.media[].assetId or nested.assetId
        media_urls: list[str] = []
        content = nested.get("content", {}) or {}
        media_list = content.get("media", []) or nested.get("media", []) or []
        for m in media_list:
            asset_id = m.get("assetId") if isinstance(m, dict) else None
            if asset_id:
                media_type = (m.get("type") or "").upper()
                ext = "source.mp4" if media_type in ("VIDEO", "REEL") else "source.jpg"
                media_urls.append(f"{self.CDN_BASE}/{asset_id}/{ext}")
        # fallback: top-level assetId
        if not media_urls:
            asset_id = nested.get("assetId") or payload.get("assetId")
            if asset_id:
                media_urls.append(f"{self.CDN_BASE}/{asset_id}/source.jpg")

        if not post_id:
            return

        result: VisionResult | None = None

        if media_urls:
            for url in media_urls[:3]:
                r = self._analyze_url(url, caption)
                if r is not None:
                    result = r
                    break
        elif caption:
            result = self._mapper.score_text(caption)

        if result is None:
            logger.warning("vision_no_result post_id=%s", post_id)
            return

        if not result.clusters and entity_id:
            logger.info(
                "vision_no_clusters_from_media using entity fallback post_id=%s entity=%s",
                post_id, entity_id,
            )

        self._publish_vision_scores(post_id, result)

    def _analyze_url(self, url: str, caption: str) -> VisionResult | None:
        try:
            with tempfile.TemporaryDirectory() as tmp:
                lower = url.lower().split("?")[0]
                is_video = any(lower.endswith(ext) for ext in (".mp4", ".mov", ".avi", ".webm", ".mkv"))
                ext = ".mp4" if is_video else ".jpg"
                dest = Path(tmp) / f"media{ext}"
                urllib.request.urlretrieve(url, dest)

                if is_video:
                    return self._mapper.score_video(dest, caption=caption)
                else:
                    return self._mapper.score_image(dest, caption=caption)
        except Exception as exc:
            logger.warning("vision_download_failed url=%s error=%s", url, exc)
            return None

    def _publish_vision_scores(self, post_id: str, result: VisionResult):
        """
        Publish cluster scores to Kafka instead of HTTP-POSTing to SimClusters.
        SimClusters' VisionScoreConsumer merges these into the post's cluster vector.
        Decoupled: vision doesn't need to know SimClusters is running.
        """
        message = {
            "postId":     post_id,
            "clusters":   result.clusters,
            "confidence": result.confidence,
            "phash":      result.phash,
            "sources":    result.sources,
        }
        try:
            self._producer.send(VISION_SCORES_TOPIC, value=message, key=post_id.encode())
            self._producer.flush(timeout=5)
            logger.info(
                "vision_scored post_id=%s clusters=%d confidence=%.3f topic=%s",
                post_id, len(result.clusters), result.confidence, VISION_SCORES_TOPIC,
            )
        except Exception as exc:
            logger.error("vision_publish_failed post_id=%s error=%s", post_id, exc)
