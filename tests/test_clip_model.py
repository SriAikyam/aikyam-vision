from vision.clip_model import ClipModel, _cluster_scores_from_prompt_probs


def test_clip_model_initial_cache_state():
    model = ClipModel()
    assert model._cached_prompt_cluster_ids is None
    assert model._cached_text_features is None


def test_cluster_scores_average_pooling_per_cluster():
    scores = _cluster_scores_from_prompt_probs(
        ["cluster_48", "cluster_48", "cluster_01"],
        [0.12, 0.42, 0.2],
        threshold=0.0,
        top_k=0,
    )

    # (0.12 + 0.42) / 2 = 0.27 for cluster_48 -- averaged, not summed, so clusters
    # with more prompts don't get an unfair advantage over clusters with fewer.
    assert scores == {
        "cluster_48": 0.27,
        "cluster_01": 0.2,
    }


def test_cluster_scores_average_avoids_prompt_count_bias():
    # A cluster with many mediocre-probability prompts should NOT beat a cluster
    # with a single strong-probability prompt under average pooling.
    scores = _cluster_scores_from_prompt_probs(
        ["many_prompts"] * 10 + ["few_prompts"],
        [0.05] * 10 + [0.3],
        threshold=0.0,
        top_k=0,
    )

    assert scores["few_prompts"] > scores["many_prompts"]


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
