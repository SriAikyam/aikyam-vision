from unittest.mock import MagicMock, patch
from worker.consumer import VisionWorker


@patch("worker.consumer.KafkaProducer")
@patch("worker.consumer.KafkaConsumer")
def _make_worker(mock_consumer, mock_producer):
    worker = VisionWorker()
    worker._producer = MagicMock()
    return worker


def test_resolve_media_items_builds_cdn_url_and_carries_asset_id():
    worker = _make_worker()
    payload = {
        "media": [
            {"assetId": "abc123", "assetType": "IMAGE"},
            {"assetId": "vid456", "assetType": "VIDEO"},
        ]
    }
    items = worker._resolve_media_items(payload)
    assert items == [
        ("https://cdn.shriaikyam.com/media/abc123/source.jpg", "IMAGE", "abc123"),
        ("https://cdn.shriaikyam.com/media/vid456/source.mp4", "VIDEO", "vid456"),
    ]


def test_resolve_media_items_skips_entries_without_asset_id():
    worker = _make_worker()
    payload = {"media": [{"assetType": "IMAGE"}, "not_a_dict", {"assetId": "ok1"}]}
    items = worker._resolve_media_items(payload)
    assert len(items) == 1
    assert items[0][2] == "ok1"


def test_publish_vision_scores_includes_asset_url():
    worker = _make_worker()
    from vision.cluster_mapper import VisionResult

    result = VisionResult(clusters={"shiva_iconography": 0.5}, confidence=0.5, sources=["clip"])
    worker._publish_vision_scores(
        "post_1", result, asset_id="abc123", media_type="IMAGE",
        asset_url="https://cdn.shriaikyam.com/media/abc123/source.jpg",
    )

    call_kwargs = worker._producer.send.call_args.kwargs
    message = call_kwargs["value"]
    assert message["assetId"] == "abc123"
    assert message["assetUrl"] == "https://cdn.shriaikyam.com/media/abc123/source.jpg"
    assert message["mediaType"] == "image"
