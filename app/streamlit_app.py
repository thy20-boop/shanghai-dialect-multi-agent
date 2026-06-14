from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import importlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import streamlit as st

from ganagent import learning as learning_module
from ganagent import tts as tts_module
from ganagent.agent import ShanghaiDialectAgent
from ganagent.asr_backends import load_audio_for_pipeline, make_backend
from ganagent.product import build_translation_product
from ganagent.repair import RepairEngine, load_custom_repairs_file, parse_custom_repairs


learning_module = importlib.reload(learning_module)
tts_module = importlib.reload(tts_module)
append_active_learning_items = learning_module.append_active_learning_items
read_active_learning_queue = learning_module.read_active_learning_queue
summarize_active_learning_items = learning_module.summarize_active_learning_items
DEFAULT_TTS_VOICE = tts_module.DEFAULT_TTS_VOICE
DEFAULT_WU_TTS_VOICE = tts_module.DEFAULT_WU_TTS_VOICE
DEFAULT_TTS_BACKEND = tts_module.DEFAULT_TTS_BACKEND
DEFAULT_WU_TTS_BACKEND = tts_module.DEFAULT_WU_TTS_BACKEND
DEFAULT_COSYVOICE_WU_URL = tts_module.DEFAULT_COSYVOICE_WU_URL
DEFAULT_GPT_SOVITS_URL = tts_module.DEFAULT_GPT_SOVITS_URL
DEFAULT_WU_REF_AUDIO = tts_module.DEFAULT_WU_REF_AUDIO
TTSRequest = tts_module.TTSRequest
mandarin_to_wu_text = tts_module.mandarin_to_wu_text
resolve_tts_text = tts_module.resolve_tts_text
synthesize_mp3 = tts_module.synthesize_mp3
wu_voice_notice = tts_module.wu_voice_notice

DEFAULT_SAMPLE = ROOT / "data" / "shanghai_audio" / "shanghai_000002.wav"
TRAINED_MODEL = ROOT / "outputs" / "models" / "whisper-small-shanghai-lora-full"
BASE_MODEL = "TingChen-ppmc/whisper-small-Shanghai"
DEFAULT_MODEL = os.environ.get("SHANGHAI_ASR_MODEL") or (
    str(TRAINED_MODEL) if TRAINED_MODEL.exists() else BASE_MODEL
)
DEFAULT_BACKEND = os.environ.get("SHANGHAI_ASR_BACKEND", "dolphin_multiagent")
DEFAULT_UI_WU_TTS_BACKEND = DEFAULT_WU_TTS_BACKEND
DEFAULT_LOCAL_FILES_ONLY = os.environ.get("HF_HUB_OFFLINE") == "1" or (
    DEFAULT_BACKEND in {"hybrid", "dolphin_multiagent"} and TRAINED_MODEL.exists()
)
DEFAULT_REPAIR_MEMORY = os.environ.get("SHANGHAI_REPAIR_MEMORY", "data/user_corrections.json")
DEFAULT_ACTIVE_LEARNING_LOG = os.environ.get("SHANGHAI_ACTIVE_LEARNING_LOG", "data/active_learning_queue.jsonl")
MODEL_PRESETS_BY_BACKEND = {
    "dolphin_multiagent": {
        "Dolphin 主识别 + 本地 LoRA 复核": "small.cn",
        "Dolphin base.cn + 本地 LoRA 复核": "base.cn",
        "自定义 Dolphin 主模型": "",
    },
    "hybrid": {
        "本地训练模型 + Dolphin 关键复核": DEFAULT_MODEL,
        "自定义本地主模型": "",
    },
    "whisper": {
        "本地训练模型（3700条）": DEFAULT_MODEL,
        "Whisper-Wu 公共吴语模型（GitHub/HF）": "peft:kaiwang0574/whisper-wu",
        "自定义模型": "",
    },
    "whisper_medium_wu": {
        "WenetSpeech-Wu Whisper-Medium-Wu（官方吴语专家）": "",
    },
    "funasr": {
        "SenseVoiceSmall（GitHub/ModelScope）": "iic/SenseVoiceSmall",
        "Fun-ASR-Nano（GitHub/HF，多方言）": "FunAudioLLM/Fun-ASR-Nano-2512",
        "自定义模型": "",
    },
    "dolphin": {
        "Dolphin small.cn（GitHub，中文方言）": "small.cn",
        "Dolphin base.cn（GitHub，中文方言）": "base.cn",
        "自定义模型": "",
    },
    "mock": {
        "Mock 示例": "mock",
    },
}


@st.cache_resource(show_spinner=False)
def get_agent(
    backend_name: str,
    model_name: str,
    assist_backend: str,
    assist_model: str,
    glossary_path: str,
    memory_path: str,
    local_files_only: bool,
    chunk_seconds: float,
    vad_enabled: bool,
    max_speech_region_seconds: float,
    custom_repairs_text: str,
) -> ShanghaiDialectAgent:
    repair_engine = (
        RepairEngine.from_file(ROOT / glossary_path)
        .with_custom_repairs(load_custom_repairs_file(ROOT / memory_path if memory_path else None))
        .with_custom_repairs(parse_custom_repairs(custom_repairs_text))
    )
    backend = make_backend(
        backend_name,
        model_name=model_name,
        local_files_only=local_files_only,
        chunk_seconds=chunk_seconds,
        vad_enabled=vad_enabled,
        max_speech_region_seconds=max_speech_region_seconds,
        assist_backend=assist_backend,
        assist_model=assist_model,
    )
    return ShanghaiDialectAgent(asr_backend=backend, repair_engine=repair_engine)


st.set_page_config(page_title="上海话转普通话", layout="wide")

st.title("上海话转普通话")
st.caption("上传一段上海话音频或视频，输出普通话文本，并标出需要复核的片段。")

with st.sidebar:
    with st.expander("高级设置"):
        backend_options = [
            "dolphin_multiagent",
            "whisper_medium_wu",
            "hybrid",
            "dolphin",
            "whisper",
            "funasr",
            "mock",
        ]
        backend_index = backend_options.index(DEFAULT_BACKEND) if DEFAULT_BACKEND in backend_options else 0
        backend_name = st.selectbox("识别后端", backend_options, index=backend_index)
        model_presets = MODEL_PRESETS_BY_BACKEND[backend_name]
        model_preset = st.selectbox("模型预设", list(model_presets.keys()), index=0)
        preset_model_name = model_presets[model_preset] or DEFAULT_MODEL
        model_name = st.text_input("模型", value=preset_model_name)
        assist_enabled = st.checkbox("启用开源模型辅助复核", value=False)
        assist_backend = "none"
        assist_model = ""
        if assist_enabled and backend_name not in {"hybrid", "dolphin_multiagent"}:
            assist_backend = st.selectbox(
                "辅助后端",
                ["whisper_medium_wu", "dolphin", "whisper", "funasr"],
                index=0,
            )
            assist_presets = MODEL_PRESETS_BY_BACKEND[assist_backend]
            assist_preset_names = list(assist_presets.keys())
            assist_default_index = 1 if assist_backend == "whisper" and len(assist_preset_names) > 1 else 0
            assist_preset = st.selectbox("辅助模型预设", assist_preset_names, index=assist_default_index)
            assist_default_model = assist_presets[assist_preset] or ""
            assist_model = st.text_input("辅助模型", value=assist_default_model)
        glossary_path = st.text_input("词典", value="data/examples/shanghainese_glossary.json")
        memory_path = st.text_input(
            "纠错记忆",
            value=DEFAULT_REPAIR_MEMORY,
            help="可选 JSON 文件；记录人名、地名、课程词等长期纠错，格式可以是 {\"错词\":\"正确词\"}。",
        )
        st.markdown("**语音输出 MP3**")
        generate_tts = st.checkbox("生成语音 MP3", value=False)
        tts_target_label = st.radio(
            "朗读内容",
            ["普通话结果", "吴语口语稿"],
            horizontal=True,
        )
        tts_backend = st.selectbox(
            "普通话 TTS 后端",
            ["edge", "command"],
            index=["edge", "command"].index(DEFAULT_TTS_BACKEND)
            if DEFAULT_TTS_BACKEND in {"edge", "command"}
            else 0,
        )
        wu_tts_backend = st.selectbox(
            "吴语 TTS 后端",
            ["cosyvoice_wu", "edge", "command"],
            index=["cosyvoice_wu", "edge", "command"].index(DEFAULT_UI_WU_TTS_BACKEND)
            if DEFAULT_UI_WU_TTS_BACKEND in {"cosyvoice_wu", "edge", "command"}
            else 0,
            help="推荐 cosyvoice_wu：调用 WenetSpeech-Wu 的 CosyVoice2 吴语专家。",
        )
        tts_voice = st.text_input("普通话 TTS 音色", value=DEFAULT_TTS_VOICE)
        wu_tts_voice = st.text_input("吴语 TTS 音色", value=DEFAULT_WU_TTS_VOICE)
        cosyvoice_wu_url = st.text_input("吴语生成专家 API", value=DEFAULT_COSYVOICE_WU_URL)
        wu_ref_audio = st.text_input("吴语参考音频（可选）", value="")
        wu_prompt_text = st.text_area(
            "吴语参考文本",
            value="",
            placeholder="留空则使用 WenetSpeech-Wu 官方上海话参考文本",
            height=80,
        )
        tts_command = st.text_input(
            "外部 TTS 命令模板",
            value=os.environ.get("SHANGHAI_TTS_COMMAND", ""),
            help="仅 command 后端使用；可用占位符：{text_file}、{output}、{ref_audio}、{prompt_text}。",
        )
        tts_rate = st.select_slider("语速", options=["-20%", "-10%", "+0%", "+10%", "+20%"], value="+0%")
        st.caption("吴语口语稿由 WenetSpeech-Wu CosyVoice2 专家生成，再交给识别与风险智能体复核。")
        save_active_learning = st.checkbox("保存主动学习候选", value=True)
        active_learning_log = st.text_input(
            "主动学习队列",
            value=DEFAULT_ACTIVE_LEARNING_LOG,
            help="高风险片段、候选分歧和修复样本会写入这个 JSONL 文件，后续可人工确认后再训练。",
        )
        queue_records = read_active_learning_queue(ROOT / active_learning_log)
        queue_summary = summarize_active_learning_items(queue_records)
        st.markdown("**主动学习队列概览**")
        q1, q2, q3 = st.columns(3)
        q1.metric("待复核", queue_summary.pending)
        q2.metric("可导出", queue_summary.exported_ready)
        q3.metric("总样本", queue_summary.total)
        if queue_summary.reason_counts:
            st.json(queue_summary.reason_counts)
        local_files_only = st.checkbox("只使用本地缓存模型", value=DEFAULT_LOCAL_FILES_ONLY)
        if backend_name == "dolphin_multiagent":
            st.info("当前默认使用多智能体协同：Dolphin 做主识别，官方 Whisper-Medium-Wu 做吴语复核/兜底，后续由仲裁、翻译、纠错记忆和风险检测智能体协作输出。")
        elif backend_name == "hybrid":
            st.info("当前默认使用混合模式：你的本地 Whisper-LoRA 做主识别，Dolphin 只做关键复核和短句纠错候选。")
        elif backend_name == "dolphin":
            st.info("当前默认使用 Dolphin 中文方言模型作为主识别器；首次使用会下载模型，后续走本地缓存。")
        elif backend_name == "funasr":
            st.info("FunASR/SenseVoice 当前只作为实验候选；首次使用需要额外安装依赖并下载模型。")
        if backend_name == "whisper" and model_name.startswith("peft:") and local_files_only:
            st.info("当前选择远程 LoRA 适配器；若本地未缓存，请先取消离线模式完成首次下载。")
        vad_enabled = st.checkbox("按停顿自动切分长音频", value=True)
        max_speech_region_seconds = st.slider("语音片段最大秒数", min_value=3.0, max_value=15.0, value=8.0, step=1.0)
        chunk_seconds = st.slider("固定兜底切片秒数", min_value=5.0, max_value=25.0, value=15.0, step=5.0)
        custom_repairs_text = st.text_area(
            "补充修复词",
            value="",
            placeholder="每行一个：错词=正确词\n例如：王家=王佳\n例如：车子机面=初次见面",
            help="用于当前运行的识别后修复，适合人名、地名、店名、课程词等专有词。",
            height=100,
        )

source = st.radio("音频来源", ["上传音频/视频", "使用示例音频"], horizontal=True)
uploaded = None
if source == "上传音频/视频":
    uploaded = st.file_uploader("上传 WAV/FLAC/OGG/M4A/MP3/MP4 音频或视频", type=["wav", "flac", "ogg", "m4a", "mp3", "mp4", "aac"])
else:
    if DEFAULT_SAMPLE.exists():
        st.audio(str(DEFAULT_SAMPLE))
    else:
        st.info("包内没有冒用旧方言样本，请先上传一段上海话音频。")

run_clicked = st.button("转成普通话", type="primary", use_container_width=True)

if run_clicked:
    audio_path: str | None = None
    temp_path: Path | None = None
    audio_duration: float | None = None
    if source == "使用示例音频" and DEFAULT_SAMPLE.exists():
        audio_path = str(DEFAULT_SAMPLE)
    elif uploaded is not None:
        suffix = Path(uploaded.name).suffix or ".wav"
        handle = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        handle.write(uploaded.getvalue())
        handle.close()
        temp_path = Path(handle.name)
        audio_path = str(temp_path)
    elif backend_name != "mock":
        st.warning("请先上传 WAV/FLAC/OGG/M4A/MP3/MP4 音频或视频。")
        st.stop()

    try:
        if audio_path:
            try:
                audio_input = load_audio_for_pipeline(Path(audio_path), 16000)
                if isinstance(audio_input, dict):
                    audio_duration = len(audio_input["raw"]) / audio_input["sampling_rate"]
            except Exception:
                audio_duration = None

        agent = get_agent(
            backend_name,
            model_name,
            assist_backend,
            assist_model,
            glossary_path,
            memory_path,
            local_files_only,
            chunk_seconds,
            vad_enabled,
            max_speech_region_seconds,
            custom_repairs_text,
        )
        progress = st.progress(0, text="准备识别...")
        status = st.status("正在识别并转换为普通话...", expanded=True)
        if audio_duration is not None and backend_name == "whisper":
            chunk_count = max(1, math.ceil(audio_duration / min(max_speech_region_seconds, chunk_seconds, 29.0)))
            estimate_low = max(20, round(audio_duration * 2))
            estimate_high = max(60, round(audio_duration * 5))
            if vad_enabled:
                status.write(
                    f"音频 {audio_duration:.1f} 秒，将先按停顿自动切分，最长约 {max_speech_region_seconds:.0f} 秒一段；"
                    f"CPU 预计约 {estimate_low}-{estimate_high} 秒。"
                )
            else:
                status.write(
                    f"音频 {audio_duration:.1f} 秒，将按固定 {chunk_seconds:.0f} 秒切分；"
                    f"CPU 预计约 {estimate_low}-{estimate_high} 秒。"
                )
        elif audio_duration is not None and backend_name in {"dolphin", "hybrid", "dolphin_multiagent"}:
            estimate_low = max(10, round(audio_duration * 0.8))
            estimate_high = max(40, round(audio_duration * 3))
            if backend_name == "dolphin_multiagent":
                status.write(
                    f"音频 {audio_duration:.1f} 秒，将由 Dolphin 主识别，再由本地 LoRA、仲裁、翻译和风险检测智能体协同处理；"
                    f"缓存后预计约 {estimate_low}-{estimate_high} 秒。"
                )
            elif backend_name == "hybrid":
                status.write(
                    f"音频 {audio_duration:.1f} 秒，将先用本地训练模型识别，再用 Dolphin 复核关键短板；"
                    f"缓存后预计约 {estimate_low}-{estimate_high} 秒。"
                )
            else:
                status.write(
                    f"音频 {audio_duration:.1f} 秒，将使用 Dolphin 上海地区方言模型识别；"
                    f"首次加载模型较慢，缓存后预计约 {estimate_low}-{estimate_high} 秒。"
                )

        def update_progress(value: float, message: str) -> None:
            progress.progress(int(value * 100), text=message)
            status.write(message)

        set_progress = getattr(agent.asr_backend, "set_progress_callback", None)
        if set_progress:
            set_progress(update_progress)
        try:
            result = agent.run(audio_path=audio_path)
            product = build_translation_product(result)
            active_learning_saved = (
                append_active_learning_items(ROOT / active_learning_log, result.active_learning_items)
                if save_active_learning
                else 0
            )
        finally:
            if set_progress:
                set_progress(None)
        progress.progress(100, text="识别完成")
        status.update(label="识别完成", state="complete", expanded=False)
    except Exception as exc:
        if "status" in locals():
            status.update(label="识别失败", state="error", expanded=True)
        st.error(str(exc))
    else:
        st.subheader("普通话结果")
        st.write(product.mandarin)

        if product.warning:
            st.warning(product.warning)
        if product.text_compacted:
            st.info("已压缩明显重复片段；原始识别保留在下方。")

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("识别状态", getattr(product, "status_label", product.status))
        c2.metric("修复次数", product.repair_count)
        c3.metric("可疑片段", product.suspicion_count)
        c4.metric("协作智能体", product.agent_count)
        c5.metric("学习候选", product.active_learning_count)
        c6.metric("质量评分", f"{product.quality_score:.2f}")

        if product.consensus_score is not None:
            st.caption(f"候选共识评分：{product.consensus_score:.2f}")
        if product.action_suggestion:
            st.info(product.action_suggestion)

        if generate_tts:
            tts_target_map = {
                "普通话结果": "mandarin",
                "吴语口语稿": "wuu",
            }
            tts_text = resolve_tts_text(
                tts_target_map[tts_target_label],
                product.mandarin,
            )
            st.subheader("语音输出 MP3")
            if tts_target_label == "吴语口语稿":
                st.write(tts_text)
            voice = wu_tts_voice if tts_target_label == "吴语口语稿" else tts_voice
            backend = wu_tts_backend if tts_target_label == "吴语口语稿" else tts_backend
            ref_audio = wu_ref_audio if tts_target_label == "吴语口语稿" and backend == "cosyvoice_wu" else None
            prompt_text = wu_prompt_text if tts_target_label == "吴语口语稿" and backend == "cosyvoice_wu" else ""
            notice = wu_voice_notice(voice, backend) if tts_target_label == "吴语口语稿" else None
            if notice:
                st.warning(notice)
            try:
                mp3_handle = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                mp3_path = Path(mp3_handle.name)
                mp3_handle.close()
                synthesize_mp3(
                    TTSRequest(
                        text=tts_text,
                        output_path=mp3_path,
                        voice=voice,
                        rate=tts_rate,
                        backend=backend,
                        ref_audio_path=ref_audio,
                        prompt_text=prompt_text,
                        cosyvoice_wu_url=cosyvoice_wu_url,
                        command_template=tts_command or None,
                    )
                )
                mp3_bytes = mp3_path.read_bytes()
                st.audio(mp3_bytes, format="audio/mp3")
                st.download_button(
                    "下载 MP3",
                    data=mp3_bytes,
                    file_name="shanghai_agent_output.mp3",
                    mime="audio/mpeg",
                    use_container_width=True,
                )
            except Exception as exc:
                st.error(f"语音合成失败：{exc}")
            finally:
                if "mp3_path" in locals() and mp3_path.exists():
                    mp3_path.unlink(missing_ok=True)

        if product.draft_mandarin:
            with st.expander("高风险草稿"):
                st.write(product.draft_mandarin)

        st.subheader("识别原文")
        st.write(result.transcript)

        with st.expander("修复记录"):
            st.json(result.repairs)

        with st.expander("可疑片段"):
            st.json([item.__dict__ for item in result.suspicions])

        if result.agent_trace:
            with st.expander("多智能体协作记录"):
                st.json(result.agent_trace)

        if result.active_learning_items:
            with st.expander("主动学习候选"):
                if active_learning_saved:
                    st.success(f"已写入 {active_learning_saved} 条到 {active_learning_log}")
                st.json(result.active_learning_items)

        if result.alternatives:
            with st.expander("开源候选识别"):
                st.json(result.alternatives)

        with st.expander("完整 JSON"):
            st.code(json.dumps(result.as_dict(), ensure_ascii=False, indent=2), language="json")
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink(missing_ok=True)
