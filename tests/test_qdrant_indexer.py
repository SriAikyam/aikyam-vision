import uuid
import pytest
from unittest.mock import MagicMock, patch
from worker.qdrant_indexer import QdrantIndexerWorker


def test_uuid5_point_id_generation():
    post_id = "6a603399f36c2841edd495f5"
    point_id_1 = str(uuid.uuid5(uuid.NAMESPACE_DNS, post_id))
    point_id_2 = str(uuid.uuid5(uuid.NAMESPACE_DNS, post_id))
    assert point_id_1 == point_id_2
    assert len(point_id_1) == 36


@patch("worker.qdrant_indexer.QdrantClient")
@patch("worker.qdrant_indexer.KafkaConsumer")
def test_qdrant_indexer_handle_valid_event(mock_kafka, mock_qdrant_client):
    mock_qdrant_instance = MagicMock()
    mock_qdrant_client.return_value = mock_qdrant_instance
    mock_qdrant_instance.get_collections.return_value.collections = []

    indexer = QdrantIndexerWorker()

    valid_payload = {
        "postId": "post_12345",
        "assetId": "asset_999",
        "mediaType": "image",
        "primaryDeity": "shiva",
        "categoryTypeScores": {"SACRED_DEVOTIONAL": 0.85},
        "clusters": {"shiva_iconography": 0.85},
        "confidence": 0.85,
        "moderationAction": "ALLOW",
        "flagged": False,
        "moderationReasons": [],
        "imageEmbedding": [0.1] * 512,
        "createdAt": "2026-07-24T10:00:00Z",
    }

    indexer._handle_event(valid_payload)

    mock_qdrant_instance.upsert.assert_called_once()
    call_args = mock_qdrant_instance.upsert.call_args
    assert call_args.kwargs["collection_name"] == "aikyam_vision_media"
    points = call_args.kwargs["points"]
    assert len(points) == 1
    assert points[0].payload["post_id"] == "post_12345"
    assert points[0].payload["primary_deity"] == "shiva"
    assert points[0].payload["moderation_action"] == "ALLOW"
    assert len(points[0].vector["clip_embedding"]) == 512


@patch("worker.qdrant_indexer.QdrantClient")
@patch("worker.qdrant_indexer.KafkaConsumer")
def test_qdrant_indexer_skip_invalid_embedding(mock_kafka, mock_qdrant_client):
    mock_qdrant_instance = MagicMock()
    mock_qdrant_client.return_value = mock_qdrant_instance

    indexer = QdrantIndexerWorker()

    invalid_payload = {
        "postId": "post_12345",
        "imageEmbedding": [0.1] * 10,  # Invalid dimension (10 instead of 512)
    }

    indexer._handle_event(invalid_payload)

    mock_qdrant_instance.upsert.assert_not_called()
