import json
from pathlib import Path

import numpy as np

from shanghaiagent import ShanghaiDialectAgent
from ganagent.asr_backends import MockASRBackend
from ganagent.asr_backends import WhisperTransformersBackend
from ganagent.asr_backends import ASRBackend
from ganagent.asr_backends import AssistedASRBackend
from ganagent.asr_backends import CollaborativeASRBackend
from ganagent.asr_backends import DolphinBackend
from ganagent.asr_backends import FunASRBackend
from ganagent.asr_backends import WhisperMediumWuBackend
from ganagent.asr_backends import make_backend
from ganagent.asr_backends import normalize_model_reference
from ganagent.asr_backends import parse_hotwords
from ganagent.asr_backends import strip_asr_control_tokens
from ganagent.codex_task import render_codex_answer_task
from ganagent.dialogue_manager import build_dialogue_reply
from ganagent.cli import append_tts_review_item
from ganagent.dialect import ShanghaiDialectDetector
from ganagent.learning import append_active_learning_items
from ganagent.learning import export_active_learning_manifest
from ganagent.learning import read_active_learning_queue
from ganagent.learning import render_active_learning_report
from ganagent.learning import summarize_active_learning_items
from ganagent.models import AgentResult, DialectSignal, Segment, Suspicion
from ganagent.product import build_translation_product
from ganagent.product import TranslationProduct
from ganagent.report import render_markdown_report
from ganagent.repair import RepairEngine, load_custom_repairs_file, parse_custom_repairs
from ganagent.speech_quality import extract_critical_entities, extract_expected_terms, score_spoken_answer
from ganagent import tts as tts_module
from ganagent.tts import (
    TTSRequest,
    allows_prefix_trim,
    build_gpt_sovits_payload,
    mandarin_to_wu_text,
    leading_hallucination_chars,
    load_wu_reference_experts,
    normalize_tts_text_for_clarity,
    resolve_tts_text,
    select_wu_reference_experts,
    split_gpt_sovits_text,
    split_tts_sentences,
    WuReferenceExpert,
    wu_voice_notice,
)
from ganagent.voice_clone import load_voice_clone_items, speaker_statistics
from ganagent.webqa import DuckDuckGoHTMLParser, clean_duckduckgo_url
from ganagent.wu_generation_expert import (
    SHANGHAI_GUARD_WU_NAME,
    build_wu_generation_policy,
    classify_wu_generation_task,
)


class FailingBackend(ASRBackend):
    name = "failing"

    def transcribe(self, audio_path: str | None = None) -> list[Segment]:
        raise RuntimeError("candidate unavailable")


def _dialogue_product(text: str, *, status: str = "ok") -> TranslationProduct:
    return TranslationProduct(
        mandarin=text,
        dialect_transcript=text,
        status=status,
        status_label="ok" if status == "ok" else "unreliable",
        warning=None,
        suspicion_count=0,
        repair_count=0,
    )


def _dialogue_result(text: str) -> AgentResult:
    return AgentResult(
        audio_path=None,
        dialect=DialectSignal(label="unknown", score=0.0),
        segments=[Segment(0.0, 1.0, text)],
        suspicions=[],
        transcript=text,
        mandarin_translation=text,
    )


def test_mock_agent_repairs_terms() -> None:
    agent = ShanghaiDialectAgent(asr_backend=MockASRBackend())
    result = agent.run()

    assert result.dialect.label == "shanghainese_or_wu"
    assert {"侬", "搿个", "阿拉"}.issubset(result.dialect.markers)
    assert "LoRA" in result.transcript
    assert "CER" in result.transcript
    assert "你好吗？" in result.mandarin_translation
    assert "我们的" in result.mandarin_translation
    assert result.repairs
    assert result.suspicions


def test_distinctive_single_marker_detects_shanghainese() -> None:
    signal = ShanghaiDialectDetector().detect("阿拉来聊聊金融方面。")
    wu_signal = ShanghaiDialectDetector().detect("吾已经做了八七年了。")

    assert signal.label == "shanghainese_or_wu"
    assert signal.score == 0.25
    assert wu_signal.label == "shanghainese_or_wu"
    assert wu_signal.score == 0.25


def test_whisper_chunk_size_stays_below_long_form_boundary() -> None:
    backend = WhisperTransformersBackend(chunk_seconds=30.0)

    assert backend._effective_chunk_seconds() == 29.0


def test_remote_whisper_wu_is_loaded_as_peft_adapter() -> None:
    assert normalize_model_reference("peft:kaiwang0574/whisper-wu") == (
        "kaiwang0574/whisper-wu",
        True,
    )
    assert normalize_model_reference("kaiwang0574/whisper-wu") == (
        "kaiwang0574/whisper-wu",
        True,
    )
    assert normalize_model_reference("openai/whisper-small") == ("openai/whisper-small", False)


def test_hybrid_backend_keeps_whisper_primary_and_dolphin_assistant() -> None:
    backend = make_backend("hybrid", model_name="outputs/models/whisper-small-shanghai-lora-full")

    assert isinstance(backend, AssistedASRBackend)
    assert isinstance(backend.primary, WhisperTransformersBackend)
    assert isinstance(backend.assistants[0], DolphinBackend)


def test_dolphin_multiagent_uses_dolphin_primary_and_whisper_medium_wu_reviewer() -> None:
    backend = make_backend("dolphin_multiagent", model_name="small.cn")

    assert isinstance(backend, CollaborativeASRBackend)
    assert isinstance(backend.primary, DolphinBackend)
    assert isinstance(backend.reviewers[0], WhisperMediumWuBackend)


def test_whisper_medium_wu_chunks_audio_below_position_limit(tmp_path) -> None:
    import soundfile as sf

    audio_path = tmp_path / "long.wav"
    sf.write(audio_path, np.zeros(16000 * 50, dtype=np.float32), 16000)

    chunks = WhisperMediumWuBackend._chunk_long_wav(audio_path)
    try:
        assert len(chunks) >= 3
        assert all(sf.info(chunk).duration <= 25 for chunk in chunks)
    finally:
        for chunk in chunks:
            if chunk != audio_path:
                chunk.unlink(missing_ok=True)


def test_collaborative_backend_traces_primary_and_reviewer() -> None:
    backend = CollaborativeASRBackend(
        MockASRBackend([Segment(start=0, end=1, text="Dolphin 主输出。")]),
        [MockASRBackend([Segment(start=0, end=1, text="LoRA 复核候选。")])],
    )

    segments = backend.transcribe("fake.wav")

    assert "".join(segment.text for segment in segments) == "Dolphin 主输出。"
    assert backend.alternatives[0]["transcript"] == "LoRA 复核候选。"
    assert {item["agent"] for item in backend.agent_trace} == {"Dolphin ASR 专家", "吴语识别复核智能体"}


def test_collaborative_backend_falls_back_when_primary_fails() -> None:
    backend = CollaborativeASRBackend(
        FailingBackend(),
        [MockASRBackend([Segment(start=0, end=1, text="兜底候选。")])],
    )

    segments = backend.transcribe("fake.wav")

    assert "".join(segment.text for segment in segments) == "兜底候选。"
    assert backend.agent_trace[0]["status"] == "failed"
    assert backend.agent_trace[-1]["status"] == "ok"


def test_funasr_segments_strip_sensevoice_control_tokens() -> None:
    backend = FunASRBackend()
    result = [
        {
            "sentence_info": [
                {"start": 120, "end": 900, "text": "<|zh|><|Speech|>侬好，初次见面。"}
            ]
        }
    ]

    segments = backend._segments_from_result(result)

    assert len(segments) == 1
    assert segments[0].start == 0.12
    assert segments[0].end == 0.9
    assert segments[0].text == "侬好，初次见面。"


def test_asr_control_token_cleanup() -> None:
    assert strip_asr_control_tokens("<|zh|><|NEUTRAL|>你好") == "你好"
    assert strip_asr_control_tokens("<zh><SHANGHAI><asr><notimestamp>侬好") == "侬好"


def test_hotwords_parse_common_separators() -> None:
    assert parse_hotwords("先生，侬好;第一趟\n上海") == ["先生", "侬好", "第一趟", "上海"]
    assert parse_hotwords(None, ["侬好"]) == ["侬好"]


def test_open_source_candidate_can_repair_primary_short_phrase_error() -> None:
    primary = MockASRBackend([Segment(start=0, end=1, text="车子机面，我叫王家。")])
    assistant = MockASRBackend([Segment(start=0, end=1, text="初次见面，我叫王佳。")])
    agent = ShanghaiDialectAgent(asr_backend=AssistedASRBackend(primary, [assistant]))

    result = agent.run("fake.wav")

    assert result.transcript == "初次见面，我叫王家。"
    assert result.alternatives[0]["status"] == "ok"
    assert any(
        replacement["source"] == "open_source_asr_candidate"
        for repair in result.repairs
        for replacement in repair.get("replacements", [])
    )


def test_greeting_short_phrase_confusions_are_repaired() -> None:
    agent = ShanghaiDialectAgent(
        asr_backend=MockASRBackend(
            [Segment(start=0, end=1, text="先什么，你好，你行，你是第一趟来上海哦，是呢，我老早没来过")]
        )
    )

    result = agent.run("fake.wav")

    assert "先生" in result.transcript
    assert "侬好" in result.transcript
    assert "第一次" in result.mandarin_translation
    assert "以前没来过" in result.mandarin_translation


def test_observed_greeting_confusions_from_user_clip_are_repaired() -> None:
    agent = ShanghaiDialectAgent(
        asr_backend=MockASRBackend(
            [Segment(start=0, end=1, text="先撒，侬好，侬行，侬是第一趟来上海哦，是呃，吾老早没来过")]
        )
    )

    result = agent.run("fake.wav")

    assert result.transcript == "先生，侬好，侬好，侬是第一趟来上海哦，是呃，吾老早没来过"
    assert result.mandarin_translation == "先生，你好，你好，你是第一次来上海哦，是的，我以前没来过"


def test_assisted_backend_keeps_primary_result_when_candidate_fails() -> None:
    backend = AssistedASRBackend(
        MockASRBackend([Segment(start=0, end=1, text="阿拉来试试看。")]),
        [FailingBackend()],
    )

    segments = backend.transcribe("fake.wav")

    assert "".join(segment.text for segment in segments) == "阿拉来试试看。"
    assert backend.alternatives[0]["status"] == "failed"


def test_agent_returns_multi_agent_trace_and_active_learning_items() -> None:
    backend = CollaborativeASRBackend(
        MockASRBackend([Segment(start=0, end=1, text="阿拉两个拧来聊聊金融方面呢")]),
        [MockASRBackend([Segment(start=0, end=1, text="我们两个人来聊聊金融方面。")])],
    )
    agent = ShanghaiDialectAgent(asr_backend=backend)

    result = agent.run("fake.wav")

    assert any(item["agent"] == "候选仲裁智能体" for item in result.agent_trace)
    assert result.active_learning_items
    assert "asr_candidate_disagreement" in result.active_learning_items[0]["reason"]


def test_candidate_rerank_replaces_repetition_hallucination() -> None:
    primary = MockASRBackend([Segment(start=0, end=8, text="现在开始" + "迁" * 80 + "到现在")])
    assistant = MockASRBackend([Segment(start=0, end=8, text="现在开始讲上海话，到现在。")])
    agent = ShanghaiDialectAgent(asr_backend=AssistedASRBackend(primary, [assistant]))

    result = agent.run("fake.wav")
    product = build_translation_product(result)

    assert result.transcript == "现在开始讲上海话，到现在。"
    assert any(item.get("type") == "alternative_rerank" for item in result.repairs)
    assert product.status != "unreliable"


def test_candidate_rerank_keeps_clean_primary_as_self_owned_result() -> None:
    primary = MockASRBackend([Segment(start=0, end=1, text="阿拉两个人来聊聊金融方面呢")])
    assistant = MockASRBackend([Segment(start=0, end=1, text="我们两个人来聊聊金融方面。")])
    agent = ShanghaiDialectAgent(asr_backend=AssistedASRBackend(primary, [assistant]))

    result = agent.run("fake.wav")

    assert result.transcript == "阿拉两个人来聊聊金融方面呢"
    assert not any(item.get("type") == "alternative_rerank" for item in result.repairs)


def test_vad_splits_clean_long_audio_on_pauses() -> None:
    sampling_rate = 16000
    silence = np.zeros(int(0.9 * sampling_rate), dtype=np.float32)
    long_pause = np.zeros(int(2.1 * sampling_rate), dtype=np.float32)
    t1 = np.linspace(0, 1.1, int(1.1 * sampling_rate), endpoint=False)
    t2 = np.linspace(0, 1.3, int(1.3 * sampling_rate), endpoint=False)
    speech_1 = (0.08 * np.sin(2 * np.pi * 220 * t1)).astype(np.float32)
    speech_2 = (0.08 * np.sin(2 * np.pi * 330 * t2)).astype(np.float32)
    raw = np.concatenate([silence, speech_1, long_pause, speech_2, silence])
    backend = WhisperTransformersBackend(chunk_seconds=15.0, max_speech_region_seconds=8.0)

    regions = backend._speech_regions(raw, sampling_rate)

    assert len(regions) == 2
    assert regions[0][0] / sampling_rate < 1.0
    assert 1.8 < regions[0][1] / sampling_rate < 2.4
    assert 3.7 < regions[1][0] / sampling_rate < 4.4


def test_context_repair_keeps_domain_specific_terms() -> None:
    segment = Segment(start=0, end=1, text="上海方言属于无语。")
    engine = RepairEngine()

    repairs = engine.repair_segments([segment])

    assert repairs
    assert segment.display_text() == "上海方言属于吴语。"


def test_segment_display_text_allows_empty_repair() -> None:
    segment = Segment(start=0, end=1, text="原文", repaired_text="")

    assert segment.display_text() == ""


def test_custom_repairs_can_be_added_for_video_specific_terms() -> None:
    repairs = parse_custom_repairs("车子机面=初次见面\n王家=>王佳")
    segment = Segment(start=0, end=1, text="车子机面，我叫王家。")
    engine = RepairEngine().with_custom_repairs(repairs)

    repair_records = engine.repair_segments([segment])

    assert segment.display_text() == "初次见面，我叫王佳。"
    assert repair_records
    assert {item["from"] for item in repair_records[0]["replacements"]} == {"车子机面", "王家"}


def test_persistent_repair_memory_file_can_be_loaded(tmp_path) -> None:
    memory = tmp_path / "memory.json"
    memory.write_text('{"王家": "王佳"}', encoding="utf-8")

    segment = Segment(start=0, end=1, text="我叫王家。")
    engine = RepairEngine().with_custom_repairs(load_custom_repairs_file(memory))

    engine.repair_segments([segment])

    assert segment.display_text() == "我叫王佳。"


def test_dialect_residue_is_translated_and_flagged_for_review() -> None:
    segment = Segment(start=0, end=1, text="阿拉两个拧来聊聊金融方面呢")
    engine = RepairEngine()

    suspicions = engine.find_suspicions([segment], repairs=[])
    translation, repairs = engine.translate_to_mandarin_with_replacements(segment.text)

    assert translation == "我们两个人来聊聊金融方面呢"
    assert repairs
    assert repairs[0]["type"] == "dialect_translation"
    assert {"from": "拧来", "to": "人来", "count": 1, "source": "translation_rules"} in repairs[0]["replacements"]
    assert any(item.reason == "dialect_review_term" and "拧来" in item.evidence for item in suspicions)


def test_agent_reports_dialect_translation_repairs() -> None:
    agent = ShanghaiDialectAgent(
        asr_backend=MockASRBackend([Segment(start=0, end=1, text="阿拉两个拧来聊聊金融方面呢")])
    )

    result = agent.run()
    product = build_translation_product(result)

    assert result.mandarin_translation == "我们两个人来聊聊金融方面呢"
    assert any(item.get("type") == "dialect_translation" for item in result.repairs)
    assert product.repair_count == 2


def test_menu_context_repairs_food_terms_across_segments() -> None:
    segments = [
        Segment(start=0, end=5, text="小龙包来一客。"),
        Segment(start=5, end=10, text="生煎慢头也要一客。"),
    ]
    engine = RepairEngine()

    repairs = engine.repair_segments(segments)

    assert len(repairs) == 2
    assert segments[0].display_text() == "小笼包来一客。"
    assert segments[1].display_text() == "生煎馒头也要一客。"


def test_replacement_character_is_high_risk() -> None:
    engine = RepairEngine()
    suspicions = engine.find_suspicions([Segment(start=0, end=1, text="这段有�字符")], repairs=[])

    assert any(item.reason == "replacement_character" and item.severity == "high" for item in suspicions)


def test_repeated_asr_hallucination_is_compacted_before_risk_check() -> None:
    segment = Segment(start=0, end=10, text="现在开始" + "迁" * 80 + "到现在")
    engine = RepairEngine()

    repairs = engine.repair_segments([segment])
    suspicions = engine.find_suspicions([segment], repairs)

    assert "[重复片段省略]" in segment.display_text()
    assert any(item["replacements"][0]["source"] == "repetition_compaction" for item in repairs)
    assert not any(item.reason == "repetition" and item.severity == "high" for item in suspicions)


def test_translation_product_compacts_repetition() -> None:
    text = "嘎一份咯，" * 20 + "�"
    result = AgentResult(
        audio_path=None,
        dialect=DialectSignal(label="shanghainese_or_wu", score=1.0),
        segments=[Segment(start=0, end=1, text=text)],
        suspicions=[
            Suspicion(
                segment_index=0,
                severity="high",
                reason="repetition",
                evidence=text,
            )
        ],
        transcript=text,
        mandarin_translation=text,
    )

    product = build_translation_product(result)

    assert product.status == "unreliable"
    assert product.text_compacted
    assert "无法可靠识别" in product.mandarin
    assert product.draft_mandarin is not None
    assert "[重复片段省略]" in product.draft_mandarin
    assert "\ufffd" not in product.mandarin


def test_translation_product_exposes_quality_and_consensus() -> None:
    result = AgentResult(
        audio_path="fake.wav",
        dialect=DialectSignal(label="shanghainese_or_wu", score=1.0),
        segments=[Segment(start=0, end=1, text="阿拉来试试看。")],
        suspicions=[],
        transcript="阿拉来试试看。",
        mandarin_translation="我们来试试看。",
        alternatives=[{"backend": "reviewer", "status": "ok", "transcript": "阿拉来试试看。"}],
        agent_trace=[{"agent": "候选仲裁智能体", "status": "primary_kept"}],
    )

    product = build_translation_product(result)

    assert product.quality_score == 1.0
    assert product.consensus_score == 1.0
    assert product.action_suggestion == "结果可直接使用，也可作为高质量样本加入展示集。"


def test_mandarin_to_wu_text_rewrites_common_phrases() -> None:
    text = mandarin_to_wu_text("你好吗？我们今天在这里等一会儿。")

    assert text == "侬好伐？阿拉今朝辣搿搭等歇。"


def test_mandarin_to_wu_text_uses_corpus_style_for_help_answer() -> None:
    text = mandarin_to_wu_text("在上海，最重要、平常最好记牢的求助电话有这几个：真正遇到生命安全危险时，不要只打12345。")

    assert "辣海上海" in text
    assert "顶顶要紧" in text
    assert "平常辰光" in text
    assert "迭几个" in text
    assert "碰着生命安全危险个辰光" in text
    assert "勿要只拨12345" in text


def test_mandarin_to_wu_text_avoids_person_word_overrewrite() -> None:
    text = mandarin_to_wu_text("请本人尽快办理，防止被他人冒用。")

    assert "请侬本人" in text
    assert "别人冒用" in text
    assert "本拧" not in text
    assert "别拧" not in text


def test_native_wu_backend_keeps_semantic_text_for_acoustic_dialect_rendering() -> None:
    source = "你在上海的话，可以申请补领身份证。"

    assert resolve_tts_text("wuu", source, backend="cosyvoice_wu") == source
    assert resolve_tts_text("wuu", source, backend="edge") != source


def test_reference_experts_prioritize_matching_domain_and_gender(tmp_path) -> None:
    female = tmp_path / "female.wav"
    male = tmp_path / "male.wav"
    female.write_bytes(b"audio")
    male.write_bytes(b"audio")
    manifest = tmp_path / "experts.json"
    manifest.write_text(
        json.dumps(
            {
                "experts": [
                    {
                        "id": "female_service",
                        "audio": str(female),
                        "prompt_text": "参考文本一",
                        "gender": "female",
                        "domains": ["public_service"],
                        "quality": 0.8,
                    },
                    {
                        "id": "male_general",
                        "audio": str(male),
                        "prompt_text": "参考文本二",
                        "gender": "male",
                        "domains": ["general"],
                        "quality": 1.0,
                    },
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    experts = load_wu_reference_experts(manifest)
    selected = select_wu_reference_experts("身份证丢失怎么办", experts, gender="auto")

    assert [item.expert_id for item in selected] == ["female_service"]


def test_leading_hallucination_detection_is_conservative() -> None:
    assert leading_hallucination_chars("居民身份证丢失", "我有居民身份证丢失") == 2
    assert leading_hallucination_chars("居民身份证丢失", "居民身份证丢失") == 0
    assert leading_hallucination_chars("请记牢", "请记牢这个号码") == 0


def test_prefix_trim_is_disabled_for_hotline_answers() -> None:
    assert allows_prefix_trim("居民身份证丢失后去派出所。")
    assert not allows_prefix_trim("报警电话110。")
    assert not allows_prefix_trim("请拨市民服务热线。")


def test_shanghai_guard_wu_hotline_policy_keeps_certified_reference_only() -> None:
    hotline = WuReferenceExpert(
        expert_id="certified_hotline",
        audio=None,
        prompt_text="一一零是报警电话。",
        domains=("hotline",),
        quality=1.0,
        use_server_default=True,
    )
    general = WuReferenceExpert(
        expert_id="daily_chat",
        audio=None,
        prompt_text="侬好。",
        domains=("general",),
        quality=0.9,
        use_server_default=True,
    )

    policy = build_wu_generation_policy(
        "请拨打110报警电话。",
        reference_experts=[general, hotline],
        endpoints=[("primary", "http://127.0.0.1:9880")],
    )

    assert policy.expert_name == SHANGHAI_GUARD_WU_NAME
    assert policy.task_type == "hotline"
    assert not policy.prefix_trim_enabled
    assert [expert.expert_id for expert in policy.reference_experts] == ["certified_hotline"]
    assert {profile.reference.expert_id for profile in policy.schedule if profile.reference} == {
        "certified_hotline"
    }


def test_shanghai_guard_wu_public_service_policy_allows_reference_exploration_and_trim() -> None:
    service = WuReferenceExpert(
        expert_id="female_service",
        audio=None,
        prompt_text="身份证丢失后去派出所。",
        domains=("public_service",),
        quality=1.0,
        use_server_default=True,
    )
    general = WuReferenceExpert(
        expert_id="male_general",
        audio=None,
        prompt_text="侬好。",
        domains=("general",),
        quality=0.8,
        use_server_default=True,
    )

    policy = build_wu_generation_policy(
        "身份证丢失后去派出所办理。",
        reference_experts=[service, general],
        endpoints=[
            ("primary", "http://127.0.0.1:9880"),
            ("secondary", "http://127.0.0.1:9881"),
        ],
    )

    assert policy.task_type == "public_service"
    assert policy.prefix_trim_enabled
    assert [expert.expert_id for expert in policy.reference_experts] == [
        "female_service",
        "male_general",
    ]
    assert any(profile.reference is general for profile in policy.schedule)
    assert any(profile.generator == "secondary" for profile in policy.schedule)


def test_generation_schedule_preserves_second_baseline_seed_for_hotlines() -> None:
    policy = build_wu_generation_policy(
        "市民服务热线是12345。",
        reference_experts=[],
        endpoints=[("primary", "http://127.0.0.1:9880")],
    )

    assert classify_wu_generation_task("市民服务热线是12345。") == "hotline"
    assert [profile.seed for profile in policy.schedule[:2]] == [1986, 2026]
    assert policy.minimum_reference_exploration == 1


def test_live_dialogue_answers_common_identity_question_locally() -> None:
    product = _dialogue_product("身份证丢了怎么办")
    reply = build_dialogue_reply(product, _dialogue_result(product.mandarin))

    assert reply.source == "local_service_rules"
    assert "派出所" in reply.text
    assert "一二三四五" in reply.text
    assert not reply.needs_codex_search


def test_live_dialogue_routes_latest_questions_to_codex_task() -> None:
    product = _dialogue_product("今天上海天气怎么样")
    reply = build_dialogue_reply(product, _dialogue_result(product.mandarin))

    assert reply.source == "codex_search_task"
    assert reply.needs_codex_search
    assert "Codex" in reply.text


def test_live_dialogue_asks_for_retry_on_unreliable_recognition() -> None:
    product = _dialogue_product("听不清", status="unreliable")
    reply = build_dialogue_reply(product, _dialogue_result(product.mandarin))

    assert reply.source == "risk_fallback"
    assert "再" in reply.text


def test_wu_voice_notice_discloses_mandarin_voice_fallback() -> None:
    assert wu_voice_notice("zh-CN-XiaoxiaoNeural", backend="edge")
    assert wu_voice_notice("wuu-CN-XiaotongNeural") is None
    assert wu_voice_notice("zh-CN-XiaoxiaoNeural", backend="gpt_sovits") is None
    assert wu_voice_notice("zh-CN-XiaoxiaoNeural", backend="cosyvoice_wu") is None


def test_gpt_sovits_payload_uses_reference_audio() -> None:
    payload = build_gpt_sovits_payload(
        TTSRequest(
            text="侬好，阿拉来试试看。",
            output_path="out.wav",
            backend="gpt_sovits",
            ref_audio_path="ref.wav",
            prompt_text="侬好伐？",
        ),
        output_path=Path("out.wav"),
    )

    assert payload["text"] == "侬好，阿拉来试试看。"
    assert payload["ref_audio_path"] == str(Path("ref.wav").resolve())
    assert payload["prompt_text"] == "侬好伐？"
    assert payload["media_type"] == "wav"
    assert payload["text_split_method"] == "cut0"
    assert payload["speed_factor"] < 1.0


def test_tts_clarity_normalization_reads_hotlines_digit_by_digit() -> None:
    text = normalize_tts_text_for_clarity("报警110，火警119，政府服务12345。")

    assert "一，一，零" in text
    assert "一，一，九" in text
    assert "一，二，三，四，五" in text


def test_gpt_sovits_text_is_split_into_short_chunks() -> None:
    text = normalize_tts_text_for_clarity(
        "辣海上海，顶顶要紧、平常辰光最好记牢个求助电话号码有迭几个：110 是报警电话；119 是火警搭消防救援电话。"
    )
    chunks = split_gpt_sovits_text(text, max_chars=28)

    assert len(chunks) > 1
    assert all(len(chunk) <= 28 for chunk in chunks)


def test_tts_sentences_split_long_answer_into_independent_clauses() -> None:
    chunks = split_tts_sentences(
        "一一零是报警电话；一一九是火警电话；一二零是医疗急救电话。",
        max_chars=18,
    )

    assert len(chunks) == 3
    assert all(len(chunk) <= 18 for chunk in chunks)


def test_tts_sentence_split_never_breaks_spoken_phone_number() -> None:
    chunks = split_tts_sentences(
        "咨询、投诉、求助搭反映勿紧急个问题，拨一，二，三，四，五。",
        max_chars=28,
    )

    assert any("一，二，三，四，五" in chunk for chunk in chunks)
    assert not any(chunk.strip("。") == "五" for chunk in chunks)


def test_critical_entities_include_phone_numbers_and_identity_terms() -> None:
    entities = extract_critical_entities("身份证丢失请去派出所，咨询电话12345。")

    assert entities == ["12345", "身份证", "派出所"]


def test_critical_entities_restore_comma_separated_spoken_digits() -> None:
    entities = extract_critical_entities("报警一，一，零。火警一，一，九。")

    assert entities == ["110", "119", "报警", "火警"]


def test_missing_critical_entity_fails_hard_gate_even_with_high_keyword_recall() -> None:
    score = score_spoken_answer(
        "报警110，火警119，急救120，交通事故122。",
        "报警110，火警119，急救120，交通事故。",
    )

    assert score.keyword_recall >= 0.8
    assert score.missing_critical_terms == ["122"]
    assert not score.passes_critical_gate
    assert not score.is_usable


def test_speech_quality_extracts_hotline_terms() -> None:
    terms = extract_expected_terms("报警110，火警119，急救120，交通事故122，政府服务12345。")

    assert terms[:5] == ["110", "119", "120", "122", "12345"]
    assert "报警" in terms
    assert "火警" in terms
    assert "急救" in terms
    assert "交通事故" in terms


def test_speech_quality_extracts_chinese_digit_hotlines() -> None:
    terms = extract_expected_terms("一一零，报警。一一九，火警。一二零，急救。")

    assert "110" in terms
    assert "119" in terms
    assert "120" in terms


def test_speech_quality_penalizes_missing_numbers() -> None:
    score = score_spoken_answer(
        "报警110，火警119，急救120，交通事故122，政府服务12345。",
        "报警电话，火警电话，急救电话，交通事故报警电话。",
    )

    assert score.keyword_recall < 0.6
    assert "110" in score.missing_terms
    assert "12345" in score.missing_terms


def test_speech_quality_accepts_spoken_digit_sequence() -> None:
    score = score_spoken_answer("报警110。", "报警一一零。")

    assert score.keyword_recall == 1.0
    assert score.is_usable


def test_speech_quality_accepts_liang_in_wu_spoken_hotline_numbers() -> None:
    score = score_spoken_answer(
        "急救电话120，服务热线12345。",
        "急救电话幺两零，服务热线幺两三四五。",
    )

    assert score.keyword_recall == 1.0
    assert score.passes_critical_gate
    assert score.char_accuracy == 1.0


def test_speech_quality_rejects_hotline_with_extra_leading_digit() -> None:
    score = score_spoken_answer("非紧急问题拨12345。", "非紧急问题拨八幺二三四五。")

    assert score.keyword_recall == 0.0
    assert score.missing_critical_terms == ["12345"]
    assert not score.passes_critical_gate


def test_speech_quality_repairs_huojing_homophone_only_with_119_context() -> None:
    contextual = score_spoken_answer("火警119。", "沪警幺幺九。")
    unrelated = score_spoken_answer("火警。", "沪警值班。")

    assert contextual.passes_critical_gate
    assert contextual.keyword_recall == 1.0
    assert not unrelated.passes_critical_gate


def test_append_tts_review_item_writes_jsonl(tmp_path) -> None:
    path = tmp_path / "tts_queue.jsonl"
    report = {
        "output": "final.mp3",
        "selected": {
            "path": "candidate.mp3",
            "text_lang": "zh",
            "quality": {
                "keyword_recall": 0.3,
                "dialect_score": 0.8,
                "missing_terms": ["110"],
                "matched_terms": ["报警"],
            },
            "transcript": "报警电话",
        },
    }

    saved = append_tts_review_item(path, report, source_text="报警110。", fallback_text="报警110。")
    row = json.loads(path.read_text(encoding="utf-8"))

    assert saved == 1
    assert row["type"] == "tts_quality_review"
    assert row["missing_terms"] == ["110"]
    assert row["keyword_recall"] == 0.3


def test_gpt_sovits_payload_uses_default_wu_reference(tmp_path, monkeypatch) -> None:
    ref_audio = tmp_path / "ref.wav"
    ref_audio.write_bytes(b"fake wav")
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("吾讲侬拔吾。", encoding="utf-8")
    monkeypatch.setattr(tts_module, "DEFAULT_WU_REF_AUDIO", str(ref_audio))
    monkeypatch.setattr(tts_module, "DEFAULT_WU_REF_PROMPT_FILE", str(prompt_file))

    payload = tts_module.build_gpt_sovits_payload(
        TTSRequest(text="侬好，阿拉来试试看。", output_path="out.wav", backend="gpt_sovits"),
        output_path=Path("out.wav"),
    )

    assert payload["ref_audio_path"] == str(ref_audio.resolve())
    assert payload["prompt_text"] == "吾讲侬拔吾。"


def test_voice_clone_manifest_statistics_load_dataset() -> None:
    items = load_voice_clone_items()
    stats = speaker_statistics(items)

    assert len(items) >= 3000
    assert stats
    assert stats[0].count > 100
    assert stats[0].duration > 60


def test_codex_answer_task_contains_handoff_requirements() -> None:
    agent = ShanghaiDialectAgent(asr_backend=MockASRBackend())
    result = agent.run("fake.wav")
    product = build_translation_product(result)

    task = render_codex_answer_task(product, result, audio_path="fake.wav")

    assert "Codex 联网问答任务" in task
    assert product.mandarin in task
    assert "Codex 执行要求" in task
    assert "联网搜索" in task
    assert "outputs\\codex_answer.mp3" in task


def test_web_search_parser_extracts_results() -> None:
    parser = DuckDuckGoHTMLParser()
    parser.feed(
        """
        <a class="result__a" href="/l/?uddg=https%3A%2F%2Fexample.com%2Fa">Example Title</a>
        <a class="result__snippet">Example snippet text.</a>
        """
    )

    assert len(parser.results) == 1
    assert parser.results[0].title == "Example Title"
    assert parser.results[0].snippet == "Example snippet text."
    assert clean_duckduckgo_url(parser.results[0].url) == "https://example.com/a"


def test_active_learning_queue_deduplicates(tmp_path) -> None:
    queue = tmp_path / "active_learning.jsonl"
    item = {
        "audio_path": "fake.wav",
        "reason": ["asr_candidate_disagreement"],
        "primary_transcript": "阿拉来试试看。",
        "candidate_transcripts": ["我们来试试看。"],
    }

    assert append_active_learning_items(queue, [item]) == 1
    assert append_active_learning_items(queue, [item]) == 0

    rows = [json.loads(line) for line in queue.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["status"] == "needs_human_review"
    assert rows[0]["id"]


def test_active_learning_summary_report_and_manifest_export(tmp_path) -> None:
    queue = tmp_path / "queue.jsonl"
    queue.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "confirmed",
                        "audio_path": "a.wav",
                        "status": "confirmed",
                        "reason": ["asr_candidate_disagreement"],
                        "primary_transcript": "错文本",
                        "confirmed_transcript": "正确文本",
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "id": "pending",
                        "audio_path": "b.wav",
                        "status": "needs_human_review",
                        "reason": ["post_asr_repair"],
                        "primary_transcript": "待确认文本",
                    },
                    ensure_ascii=False,
                ),
                "{bad json",
            ]
        ),
        encoding="utf-8-sig",
    )

    records = read_active_learning_queue(queue)
    summary = summarize_active_learning_items(records)
    report = render_active_learning_report(records)
    manifest = tmp_path / "manifest.jsonl"
    exported = export_active_learning_manifest(queue, manifest)

    assert len(records) == 2
    assert summary.total == 2
    assert summary.pending == 1
    assert summary.confirmed == 1
    assert summary.exported_ready == 1
    assert summary.reason_counts["asr_candidate_disagreement"] == 1
    assert "正确文本" in report
    assert exported == 1
    row = json.loads(manifest.read_text(encoding="utf-8").strip())
    assert row["audio"] == "a.wav"
    assert row["text"] == "正确文本"


def test_markdown_report_includes_multi_agent_sections() -> None:
    result = AgentResult(
        audio_path="fake.wav",
        dialect=DialectSignal(label="shanghainese_or_wu", score=1.0),
        segments=[Segment(start=0, end=1, text="阿拉来试试看。")],
        suspicions=[],
        transcript="阿拉来试试看。",
        mandarin_translation="我们来试试看。",
        repairs=[
            {
                "type": "dialect_translation",
                "original": "阿拉来试试看。",
                "repaired": "我们来试试看。",
            }
        ],
        alternatives=[{"backend": "whisper", "status": "ok", "transcript": "我们来试试看。"}],
        agent_trace=[{"agent": "候选仲裁智能体", "role": "多候选重排", "status": "primary_kept"}],
        active_learning_items=[
            {
                "reason": ["asr_candidate_disagreement"],
                "primary_transcript": "阿拉来试试看。",
                "candidate_transcripts": ["我们来试试看。"],
            }
        ],
    )

    report = render_markdown_report(result)

    assert "## Multi-Agent Trace" in report
    assert "## ASR Alternatives" in report
    assert "## Active Learning Candidates" in report
    assert "## Quality Assessment" in report
    assert "`dialect_translation` segment all" in report
