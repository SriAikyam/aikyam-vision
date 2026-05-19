import pytest

from vision.cluster_mapper import ClusterMapper, _keyword_score, _merge_scores
from vision.exif_extractor import ExifResult


def test_keyword_score_shiva():
    scores = _keyword_score("Om Namah Shivaya from Kedarnath temple")
    assert "cluster_01" in scores
    assert scores["cluster_01"] > 0.0


def test_keyword_score_krishna():
    scores = _keyword_score("Hare Krishna devotees in Vrindavan dancing kirtan")
    assert "cluster_03" in scores
    assert scores["cluster_03"] >= 0.9


def test_keyword_score_empty():
    assert _keyword_score("") == {}
    assert _keyword_score("   ") == {}


def test_keyword_score_no_match():
    scores = _keyword_score("random text about cats and dogs")
    assert scores == {}


def test_keyword_score_festival():
    scores = _keyword_score("Happy Diwali celebration diyas")
    assert "cluster_55" in scores


def test_merge_scores_probabilistic_or():
    exif = ExifResult()
    merged = _merge_scores(
        clip={"cluster_01": 0.6},
        keywords={"cluster_01": 0.5},
        exif=exif,
    )
    # 1 - (1-0.6)*(1-0.5) = 1 - 0.4*0.5 = 0.8
    assert abs(merged["cluster_01"] - 0.8) < 0.01


def test_merge_scores_gps_boost():
    exif = ExifResult(
        gps_lat=25.31, gps_lon=83.01, sacred_cluster="cluster_01", sacred_score=0.9
    )
    merged = _merge_scores(clip={}, keywords={}, exif=exif)
    assert "cluster_01" in merged
    assert merged["cluster_01"] >= 0.9


def test_score_text_mapper():
    mapper = ClusterMapper()
    result = mapper.score_text("Beautiful Ganesh Chaturthi celebration Siddhivinayak")
    assert "cluster_48" in result.clusters
    assert result.confidence > 0.0
    assert "keywords" in result.sources


def test_score_text_empty():
    mapper = ClusterMapper()
    result = mapper.score_text("")
    assert result.clusters == {}
    assert result.confidence == 0.0


def test_score_text_multiple_clusters():
    mapper = ClusterMapper()
    result = mapper.score_text("Bhajan kirtan at Shiva temple during Diwali puja aarti")
    assert len(result.clusters) >= 2
