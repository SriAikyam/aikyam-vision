from vision.clip_model import _cluster_scores_from_prompt_probs


def test_cluster_scores_keep_best_prompt_per_cluster():
    scores = _cluster_scores_from_prompt_probs(
        ["cluster_48", "cluster_48", "cluster_01"],
        [0.12, 0.42, 0.2],
        threshold=0.0,
        top_k=0,
    )

    assert scores == {
        "cluster_48": 0.42,
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
