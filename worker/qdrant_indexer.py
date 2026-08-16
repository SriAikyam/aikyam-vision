from __future__ import annotations

import json
import logging
import time
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from kafka import KafkaConsumer

from vision.config import (
    KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_PRODUCER_BOOTSTRAP_SERVERS,
    KAFKA_AUTO_OFFSET_RESET,
    VISION_SCORES_TOPIC,
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION,
    QDRANT_CONSUMER_GROUP,
)

logger = logging.getLogger("aikyam.vision.indexer")


class QdrantIndexerWorker:
    """
    Decoupled Indexer Worker:
    Consumes rich vision events from VISION_SCORES_TOPIC and upserts
    512-D vectors with payload metadata into Qdrant Vector Database.
    """

    def __init__(self):
        if QDRANT_URL:
            logger.info(
                "qdrant_indexer_init url=%s collection=%s topic=%s",
                QDRANT_URL, QDRANT_COLLECTION, VISION_SCORES_TOPIC,
            )
            self._qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=10)
        else:
            logger.info(
                "qdrant_indexer_init host=%s port=%d collection=%s topic=%s",
                QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION, VISION_SCORES_TOPIC,
            )
            self._qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, api_key=QDRANT_API_KEY, timeout=10)
        self._ensure_collection()

        self._consumer = None
        for attempt in range(1, 6):
            try:
                self._consumer = KafkaConsumer(
                    VISION_SCORES_TOPIC,
                    bootstrap_servers=KAFKA_PRODUCER_BOOTSTRAP_SERVERS,
                    group_id=QDRANT_CONSUMER_GROUP,
                    auto_offset_reset=KAFKA_AUTO_OFFSET_RESET,
                    enable_auto_commit=True,
                    api_version=(2, 8, 0),
                    value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                )
                logger.info("kafka_indexer_consumer_connected attempt=%d", attempt)
                break
            except Exception as exc:
                logger.warning("kafka_indexer_connect_retry attempt=%d error=%s", attempt, exc)
                time.sleep(3)

        if self._consumer is None:
            raise RuntimeError("Failed to connect QdrantIndexerWorker to Kafka after 5 attempts")

    def _ensure_collection(self):
        """Ensures Qdrant collection exists with 512-D Cosine configuration."""
        try:
            collections = [c.name for c in self._qdrant.get_collections().collections]
            if QDRANT_COLLECTION not in collections:
                self._qdrant.create_collection(
                    collection_name=QDRANT_COLLECTION,
                    vectors_config={"clip_embedding": VectorParams(size=512, distance=Distance.COSINE)},
                )
                logger.info("qdrant_collection_created collection=%s size=512 distance=COSINE", QDRANT_COLLECTION)
            else:
                logger.info("qdrant_collection_exists collection=%s", QDRANT_COLLECTION)
        except Exception as exc:
            logger.error("qdrant_ensure_collection_failed error=%s", exc)

    def _handle_event(self, payload: dict):
        post_id = payload.get("postId")
        embedding = payload.get("imageEmbedding")

        if not post_id or not embedding or len(embedding) != 512:
            logger.warning("qdrant_skip_invalid_event post_id=%s has_embedding=%s", post_id, bool(embedding))
            return

        # Generate deterministic UUID for Qdrant point based on post_id
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, post_id))

        point_payload = {
            "post_id":              post_id,
            "asset_id":             payload.get("assetId", ""),
            "media_type":           payload.get("mediaType", "image"),
            "primary_deity":        payload.get("primaryDeity"),
            "category_type_scores": payload.get("categoryTypeScores", {}),
            "clusters":             payload.get("clusters", {}),
            "confidence":           payload.get("confidence", 0.0),
            "moderation_action":    payload.get("moderationAction", "ALLOW"),
            "flagged":              payload.get("flagged", False),
            "moderation_reasons":   payload.get("moderationReasons", []),
            "created_at":           payload.get("createdAt", ""),
        }

        try:
            point = PointStruct(id=point_id, vector={"clip_embedding": embedding}, payload=point_payload)
            self._qdrant.upsert(
                collection_name=QDRANT_COLLECTION,
                points=[point],
            )
            logger.info(
                "qdrant_point_upserted post_id=%s point_id=%s deity=%s action=%s",
                post_id, point_id, payload.get("primaryDeity"), payload.get("moderationAction"),
            )
        except Exception as exc:
            logger.error("qdrant_upsert_failed post_id=%s error=%s", post_id, exc)

    def start(self):
        logger.info("qdrant_indexer_started topic=%s", VISION_SCORES_TOPIC)
        for message in self._consumer:
            try:
                self._handle_event(message.value)
            except Exception as exc:
                logger.error("qdrant_indexer_handle_error error=%s", exc)
