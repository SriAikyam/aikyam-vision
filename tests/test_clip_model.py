from vision.clip_model import _cluster_scores_from_prompt_probs


def test_cluster_scores_sum_pooling_per_cluster():
    scores = _cluster_scores_from_prompt_probs(
        ["cluster_48", "cluster_48", "cluster_01"],
        [0.12, 0.42, 0.2],
        threshold=0.0,
        top_k=0,
    )

    # 0.12 + 0.42 = 0.54 for cluster_48
    assert scores == {
        "cluster_48": 0.54,
        "cluster_01": 0.2,
    }


def test_cluster_scores_apply_threshold_and_top_k():
    scores = _cluster_scores_from_prompt_probs(
        ["cluster_48", "cluster_01", "cluster_53", "cluster_54"],
        [0.42, 0.18, 0.09, 0.04],
        threshold=0.05,
        top_k=2,
    )

    assert scores == {
        "cluster_48": 0.42,
        "cluster_01": 0.18,
    }


def test_cluster_scores_include_non_spiritual_clusters():
    scores = _cluster_scores_from_prompt_probs(
        ["cluster_48", "cluster_90", "cluster_01"],
        [0.5, 0.9, 0.3],
        threshold=0.0,
        top_k=0,
    )

    # Non-spiritual clusters must be returned like normal clusters
    assert scores == {
        "cluster_90": 0.9,
        "cluster_48": 0.5,
        "cluster_01": 0.3,
    }
