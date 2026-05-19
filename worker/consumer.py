from __future__ import annotations

import json
import logging
import tempfile
import urllib.request
from pathlib import Path

import requests
from kafka import KafkaConsumer

from vision.cluster_mapper import ClusterMapper, VisionResult
from vision.config import (
    HTTP_TIMEOUT_SECONDS,
    KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_CONSUMER_GROUP,
    POST_CREATED_TOPIC,
    SIMCLUSTERS_BASE_URL,
)

logger = logging.getLogger("aikyam.vision.worker")


class VisionWorker:
    """
    Consumes post_created events, runs vision analysis, pushes cluster scores
    to aikyam-simclusters so each post has a KnownFor vector before first impression.
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
        self._mapper = ClusterMapper()

    def start(self):
        logger.info("Vision worker started, listening on %s", POST_CREATED_TOPIC)
        for message in self._consumer:
            self._handle(message.value)

    def _handle(self, payload: dict):
        post_id: str = payload.get("postId", "")
        caption: str = payload.get("caption", "")
        media_urls: list[str] = payload.get("mediaUrls", [])
        entity_id: str | None = payload.get("entityId")

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
            logger.info("vision_no_clusters_from_media using entity fallback post_id=%s entity=%s", post_id, entity_id)

        self._push_to_simclusters(post_id, result)

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
                    return self._mapper.score_image(dest)
        except Exception as exc:
            logger.warning("vision_download_failed url=%s error=%s", url, exc)
            return None

    def _push_to_simclusters(self, post_id: str, result: VisionResult):
        body = {
            "clusters": result.clusters,
            "confidence": result.confidence,
            "phash": result.phash,
            "sources": result.sources,
        }
        try:
            resp = requests.post(
                f"{SIMCLUSTERS_BASE_URL}/sim/post/{post_id}/score",
                json=body,
                timeout=HTTP_TIMEOUT_SECONDS,
            )
            resp.raise_for_status()
            logger.info(
                "vision_scored post_id=%s clusters=%d confidence=%.3f",
                post_id, len(result.clusters), result.confidence,
            )
        except Exception as exc:
            logger.error("vision_simclusters_push_failed post_id=%s error=%s", post_id, exc)
