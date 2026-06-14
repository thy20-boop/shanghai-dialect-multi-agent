from ganagent.dashboard import render_html_dashboard, summarize_predictions


def test_dashboard_contains_metrics_and_transcript() -> None:
    html = render_html_dashboard(
        [
            {
                "id": "demo",
                "transcript": "侬好伐",
                "mandarin_translation": "你好吗",
                "dialect_markers": ["侬", "伐"],
                "repair_count": 0,
                "suspicion_count": 0,
                "agent_trace": [{"agent": "候选仲裁智能体", "status": "primary_kept"}],
                "active_learning_items": [{"reason": ["post_asr_repair"]}],
            }
        ],
        {"sample_count": 1, "cer": 0.0},
    )

    assert "Shanghai Dialect ASR Dashboard" in html
    assert "侬好伐" in html
    assert "Learning Items" in html
    assert "候选仲裁智能体" in html


def test_dashboard_summarizes_multi_agent_quality_counts() -> None:
    summary = summarize_predictions(
        [
            {"repair_count": 2, "suspicion_count": 1, "active_learning_items": [{"reason": ["x"]}]},
            {"repair_count": 0, "suspicion_count": 0, "active_learning_items": []},
        ]
    )

    assert summary["avg_repairs"] == 1.0
    assert summary["avg_suspicions"] == 0.5
    assert summary["active_learning_items"] == 1
    assert summary["needs_review"] == 1
