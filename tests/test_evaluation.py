from ganagent.evaluation import cer, evaluate_pairs


def test_cer_exact_match() -> None:
    assert cer("侬好伐", "侬好伐") == 0.0


def test_evaluate_pairs_tracks_terms_and_markers() -> None:
    summary = evaluate_pairs(
        [("侬好伐 LoRA CER", "侬好伐 LoRA 塞尔")],
        domain_terms=["LoRA", "CER"],
        dialect_markers=["侬", "伐"],
    )

    assert summary.sample_count == 1
    assert summary.term_recall == 0.5
    assert summary.dialect_marker_recall == 1.0
