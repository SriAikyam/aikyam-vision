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
    KAFKA_CONSUMER_BOOTSTRAP_SERVERS,
    KAFKA_PRODUCER_BOOTSTRAP_SERVERS,
    KAFKA_CONSUMER_GROUP,
    KAFKA_AUTO_OFFSET_RESET,
    POST_CREATED_TOPIC,
    VISION_SCORES_TOPIC,
    AIKYAM_CDN_BASE_URL,
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
        import time
        from kafka.errors import NoBrokersAvailable

        logger.info("Initializing VisionWorker, connecting to Kafka...")
        for attempt in range(1, 6):
            try:
                self._consumer = KafkaConsumer(
                    POST_CREATED_TOPIC,
                    bootstrap_servers=KAFKA_CONSUMER_BOOTSTRAP_SERVERS,
                    group_id=KAFKA_CONSUMER_GROUP,
                    auto_offset_reset=KAFKA_AUTO_OFFSET_RESET,
                    enable_auto_commit=True,
                    api_version=(2, 8, 0),
                )
                self._producer = KafkaProducer(
                    bootstrap_servers=KAFKA_PRODUCER_BOOTSTRAP_SERVERS,
                    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                    acks=1,
                    retries=3,
                )
                logger.info("Successfully connected to Kafka!")
                break
            except NoBrokersAvailable as e:
                if attempt == 5:
                    logger.error("Failed to connect to Kafka after 5 attempts.")
                    raise e
                logger.warning("Kafka not ready yet (attempt %d/5), retrying in 5 seconds...", attempt)
                time.sleep(5)
        self._mapper = ClusterMapper()

    def start(self):
        logger.info("Vision worker started, listening on %s", POST_CREATED_TOPIC)
        for message in self._consumer:
            try:
                if not message.value:
                    continue
                payload = json.loads(message.value.decode("utf-8"))
                logger.info("kafka_event_consumed topic=%s post_id=%s", POST_CREATED_TOPIC, payload.get("postId") or payload.get("entityId"))
                self._handle(payload)
            except Exception as exc:
                logger.error("Failed to deserialize or process message: %s", exc)

    def _resolve_media_items(self, payload: dict) -> list[tuple[str, str, str]]:
        media_items = []
        media_list = payload.get("media", []) or []
        for m in media_list:
            if not isinstance(m, dict):
                continue
            asset_id = m.get("assetId")
            if asset_id:
                asset_type = (m.get("assetType") or "IMAGE").upper()
                ext = "source.mp4" if asset_type in ("VIDEO", "REEL") else "source.jpg"
                url = f"{AIKYAM_CDN_BASE_URL}/{asset_id}/{ext}"
                media_items.append((url, asset_type, asset_id))
        return media_items

    def _handle(self, payload: dict):
        post_id: str = payload.get("postId", "") or payload.get("entityId", "")

        nested   = payload.get("payload", {}) or {}
        caption: str = nested.get("postText", "") or payload.get("text", "") or ""
        entity_id: str | None = payload.get("templeId") or payload.get("entityId")

        # Resolve media items from assetId using CDN pattern
        media_items = self._resolve_media_items(payload)

        if not post_id:
            return

        result: VisionResult | None = None
        used_url = ""
        used_asset_id = ""
        used_media_type = "IMAGE"

        if media_items:
            for url, asset_type, asset_id in media_items[:3]:
                r = self._analyze_media(url, asset_type, caption)
                if r is not None:
                    result = r
                    used_url = url
                    used_asset_id = asset_id
                    used_media_type = asset_type
                    break
        elif caption:
            result = self._mapper.score_text(caption)
            used_media_type = "TEXT"

        if result is None:
            logger.warning("vision_no_result post_id=%s", post_id)
            return

        if not result.clusters and entity_id:
            logger.info(
                "vision_no_clusters_from_media using entity fallback post_id=%s entity=%s",
                post_id, entity_id,
            )

        self._publish_vision_scores(
            post_id, result, asset_id=used_asset_id, media_type=used_media_type, asset_url=used_url,
        )

    def _analyze_media(self, url: str, asset_type: str, caption: str) -> VisionResult | None:
        try:
            with tempfile.TemporaryDirectory() as tmp:
                is_video = asset_type in ("VIDEO", "REEL")
                ext = ".mp4" if is_video else ".jpg"
                dest = Path(tmp) / f"media{ext}"

                req = urllib.request.Request(
                    url,
                    headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
                    },
                )
                with urllib.request.urlopen(req) as resp, open(dest, "wb") as out:
                    out.write(resp.read())

                if is_video:
                    return self._mapper.score_video(dest, caption=caption)
                else:
                    return self._mapper.score_image(dest, caption=caption)
        except Exception as exc:
            logger.warning("vision_download_failed url=%s error=%s", url, exc)
            return None

    def _publish_vision_scores(
        self, post_id: str, result: VisionResult, asset_id: str = "", media_type: str = "IMAGE", asset_url: str = "",
    ):
        """
        Publish cluster scores and embeddings to Kafka.
        Streams full rich events containing embeddings and scores to VISION_SCORES_TOPIC.
        """
        import datetime
        message = {
            "postId":            post_id,
            "assetId":           asset_id,
            "assetUrl":          asset_url,
            "mediaType":         media_type.lower(),
            "clusters":          result.clusters,
            "confidence":        result.confidence,
            "phash":             result.phash,
            "sources":           result.sources,
            "imageEmbedding":    result.image_embedding,
            "primaryDeity":      result.primary_deity,
            "categoryTypeScores": result.category_type_scores,
            "primaryPillar":     result.primary_pillar,
            "isDevotional":      result.is_devotional,
            "flagged":           result.flagged,
            "moderationAction":  result.moderation_action,
            "moderationReasons": result.moderation_reasons,
            "createdAt":         datetime.datetime.now(datetime.timezone.utc).isoformat(),
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
