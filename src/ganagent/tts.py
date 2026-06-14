from __future__ import annotations

import asyncio
import io
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import threading
import wave
from dataclasses import dataclass, replace
from difflib import SequenceMatcher
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


DEFAULT_TTS_VOICE = os.environ.get("SHANGHAI_TTS_VOICE", "zh-CN-XiaoxiaoNeural")
DEFAULT_WU_TTS_VOICE = os.environ.get("SHANGHAI_WU_TTS_VOICE", DEFAULT_TTS_VOICE)
DEFAULT_TTS_BACKEND = os.environ.get("SHANGHAI_TTS_BACKEND", "edge")
DEFAULT_WU_TTS_BACKEND = os.environ.get("SHANGHAI_WU_TTS_BACKEND", "cosyvoice_wu")
DEFAULT_COSYVOICE_WU_URL = os.environ.get(
    "SHANGHAI_COSYVOICE_WU_URL",
    "http://127.0.0.1:9881/tts",
)
DEFAULT_GPT_SOVITS_URL = os.environ.get("SHANGHAI_GPT_SOVITS_URL", "http://127.0.0.1:9880/tts")
DEFAULT_WU_REF_AUDIO = os.environ.get(
    "SHANGHAI_WU_REF_AUDIO",
    "outputs/wu_tts_assets_multi_clean/reference/G0018_ref_01.wav",
)
DEFAULT_WU_REF_PROMPT_FILE = os.environ.get(
    "SHANGHAI_WU_REF_PROMPT_FILE",
    "outputs/wu_tts_assets_multi_clean/reference_prompt.txt",
)
DEFAULT_GPT_SOVITS_SPEED = float(os.environ.get("SHANGHAI_GPT_SOVITS_SPEED", "0.92"))
DEFAULT_GPT_SOVITS_FRAGMENT_INTERVAL = float(os.environ.get("SHANGHAI_GPT_SOVITS_FRAGMENT_INTERVAL", "0.45"))
DEFAULT_GPT_SOVITS_MAX_CHARS = int(os.environ.get("SHANGHAI_GPT_SOVITS_MAX_CHARS", "38"))
DEFAULT_WU_REFERENCE_EXPERTS = os.environ.get(
    "SHANGHAI_WU_REFERENCE_EXPERTS",
    "configs/wu_reference_experts.json",
)

MANDARIN_TO_WU_RULES = [
    ("在上海，最重要、平常最好记牢的求助电话有这几个：", "辣海上海，顶顶要紧、平常辰光最好记牢个求助电话号码有迭几个："),
    ("简单记就是：", "简单点记就是："),
    ("真正遇到生命安全危险时", "真正碰着生命安全危险个辰光"),
    ("先打紧急电话", "先拨紧急电话"),
    ("不要只打12345", "勿要只拨12345"),
    ("不要只打", "勿要只拨"),
    ("遇到治安、刑事案件或紧急危险时打", "碰着治安、刑事案件或者紧急危险个辰光拨"),
    ("火警和消防救援电话", "火警搭消防救援电话"),
    ("医疗急救电话", "医疗急救电话"),
    ("交通事故报警电话", "交通事故报警电话"),
    ("市民服务热线", "市民服务热线"),
    ("适合咨询、投诉、求助和反映非紧急问题", "适合咨询、投诉、求助搭反映勿紧急个问题"),
    ("政府服务", "政府服务"),
    ("紧急电话", "紧急电话"),
    ("求助电话", "求助电话号码"),
    ("电话号码", "电话号码"),
    ("最重要", "顶顶要紧"),
    ("平常最好记牢", "平常辰光最好记牢"),
    ("平常", "平常辰光"),
    ("这几个", "迭几个"),
    ("遇到", "碰着"),
    ("危险时", "危险个辰光"),
    ("反映", "反映"),
    ("咨询", "咨询"),
    ("投诉", "投诉"),
    ("求助", "求助"),
    ("非紧急", "勿紧急"),
    ("刚刚丢了", "刚刚落脱哉"),
    ("身份证", "身份证"),
    ("不用着急", "勿用急"),
    ("请本人", "请侬本人"),
    ("尽快", "赶紧"),
    ("就近", "就近"),
    ("防止被他人冒用", "防止拨别人冒用"),
    ("同时可以", "同时可以"),
    ("申请补领", "申请补领"),
    ("网上补领", "网上补领"),
    ("本市户籍居民", "本市户籍居民"),
    ("可以在线申请", "可以在线申请"),
    ("工本费", "工本费"),
    ("如果急需用证", "假使急等要用证"),
    ("补领期间", "补领辰光"),
    ("咨询派出所", "问派出所"),
    ("临时居民身份证", "临时居民身份证"),
    ("你好吗？", "侬好伐？"),
    ("你好吗", "侬好伐"),
    ("我听不太清楚", "吾听勿大清爽"),
    ("我们的", "阿拉个"),
    ("我们", "阿拉"),
    ("他们", "伊拉"),
    ("在上海", "辣海上海"),
    ("在什么地方", "辣撒地方"),
    ("在这里", "辣搿搭"),
    ("这里", "搿搭"),
    ("人来", "拧来"),
    ("三个人", "三家头"),
    ("几个月", "几个号头"),
    ("个月", "号头"),
    ("大约", "毛毛叫"),
    ("这个", "搿个"),
    ("今天", "今朝"),
    ("那么", "葛末"),
    ("时候", "辰光"),
    ("不要", "勿要"),
    ("怎么", "哪能"),
    ("还有", "伐有"),
    ("下次", "下趟"),
    ("第一次", "第一趟"),
    ("是的", "是额"),
    ("以前", "老早"),
    ("找个地方", "寻呃地方"),
    ("找一找", "寻一寻"),
    ("什么", "啥"),
    ("等一会儿", "等歇"),
    ("清楚", "清爽"),
    ("都", "侪"),
    ("很", "交关"),
    ("他", "伊"),
    ("我", "吾"),
    ("你", "侬"),
    ("人", "拧"),
    ("不", "勿"),
    ("。", "。"),
]


@dataclass
class TTSRequest:
    text: str
    output_path: str | Path
    voice: str = DEFAULT_TTS_VOICE
    rate: str = "+0%"
    pitch: str = "+0Hz"
    backend: str = DEFAULT_TTS_BACKEND
    ref_audio_path: str | Path | None = None
    prompt_text: str = ""
    text_lang: str = "zh"
    prompt_lang: str = "zh"
    cosyvoice_wu_url: str = DEFAULT_COSYVOICE_WU_URL
    gpt_sovits_url: str = DEFAULT_GPT_SOVITS_URL
    command_template: str | None = None
    speed: float = 1.0
    instruction: str = "这是一位上海人，用自然、清楚、平稳的上海话说"
    use_text_frontend: bool = True
    seed: int = 1986


@dataclass(frozen=True)
class WuReferenceExpert:
    expert_id: str
    audio: str | None
    prompt_text: str
    gender: str = "unknown"
    domains: tuple[str, ...] = ("general",)
    quality: float = 0.5
    use_server_default: bool = False


def load_wu_reference_experts(
    path: str | Path = DEFAULT_WU_REFERENCE_EXPERTS,
) -> list[WuReferenceExpert]:
    manifest = Path(path)
    if not manifest.exists():
        return []
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    rows = payload.get("experts", []) if isinstance(payload, dict) else payload
    experts: list[WuReferenceExpert] = []
    for row in rows:
        audio = str(row.get("audio") or "").strip()
        prompt_text = str(row.get("prompt_text") or "").strip()
        use_server_default = bool(row.get("use_server_default", False))
        if (not audio and not use_server_default) or not prompt_text or row.get("enabled", True) is False:
            continue
        audio_path = None
        if audio:
            audio_path = Path(audio)
            if not audio_path.is_absolute():
                audio_path = (manifest.parent.parent / audio_path).resolve()
            if not audio_path.exists():
                continue
        experts.append(
            WuReferenceExpert(
                expert_id=str(row.get("id") or (audio_path.stem if audio_path else "server_default")),
                audio=str(audio_path) if audio_path else None,
                prompt_text=prompt_text,
                gender=str(row.get("gender") or "unknown").lower(),
                domains=tuple(str(item) for item in row.get("domains", ["general"])),
                quality=float(row.get("quality", 0.5)),
                use_server_default=use_server_default,
            )
        )
    return experts


def select_wu_reference_experts(
    text: str,
    experts: list[WuReferenceExpert],
    *,
    gender: str = "auto",
) -> list[WuReferenceExpert]:
    if not experts:
        return []
    requested_gender = gender.strip().lower()
    filtered = [
        expert
        for expert in experts
        if requested_gender in {"", "auto", "any"}
        or expert.gender in {requested_gender, "unknown", "any"}
    ] or experts
    domains = {"general"}
    if re.search(r"\d{2,}|报警|火警|急救|热线|电话", text):
        domains.add("hotline")
    if any(term in text for term in ("身份证", "派出所", "户籍", "居民证")):
        domains.add("public_service")
    specific_domains = domains - {"general"}
    if specific_domains:
        domain_matched = [
            expert for expert in filtered if specific_domains.intersection(expert.domains)
        ]
        if domain_matched:
            filtered = domain_matched
    return sorted(
        filtered,
        key=lambda expert: (
            bool(domains.intersection(expert.domains)),
            expert.quality,
            expert.expert_id,
        ),
        reverse=True,
    )


def leading_hallucination_chars(expected_text: str, recognized_text: str) -> int:
    """Return a conservative count of ASR characters inserted before the answer."""

    expected = _normalize_alignment_text(expected_text)
    recognized = _normalize_alignment_text(recognized_text)
    if len(expected) < 4 or len(recognized) < 6:
        return 0
    matcher = SequenceMatcher(None, expected, recognized, autojunk=False)
    blocks = [block for block in matcher.get_matching_blocks() if block.size >= 3]
    if not blocks:
        return 0
    first = min(blocks, key=lambda block: (block.a, block.b))
    if first.a > 2 or first.b < 2:
        return 0
    prefix = recognized[: first.b]
    if any(prefix.endswith(expected[:size]) for size in range(1, min(3, len(expected)) + 1)):
        return 0
    return first.b


def allows_prefix_trim(text: str) -> bool:
    """Keep safety-critical hotline audio untouched after candidate generation."""

    return not (
        any(char.isdigit() for char in text)
        or any(term in text for term in ("报警", "火警", "急救", "热线", "电话"))
    )


def trim_wav_leading_hallucination(
    input_path: str | Path,
    output_path: str | Path,
    expected_text: str,
    recognized_text: str,
) -> dict | None:
    prefix_chars = leading_hallucination_chars(expected_text, recognized_text)
    if not prefix_chars:
        return None
    import numpy as np
    import soundfile as sf

    audio, sample_rate = sf.read(str(input_path), dtype="float32")
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    duration = len(audio) / float(sample_rate)
    recognized_length = len(_normalize_alignment_text(recognized_text))
    if duration <= 0 or recognized_length <= 0:
        return None
    estimated = duration * prefix_chars / recognized_length
    if estimated < 0.12 or estimated > min(1.8, duration * 0.35):
        return None
    search_radius = 0.24
    start = max(0, int((estimated - search_radius) * sample_rate))
    end = min(len(audio), int((estimated + search_radius) * sample_rate))
    window = max(1, int(0.04 * sample_rate))
    hop = max(1, int(0.01 * sample_rate))
    offsets = range(start, max(start + 1, end - window), hop)
    cut = min(
        offsets,
        key=lambda offset: float(np.mean(np.abs(audio[offset : offset + window]))),
        default=int(estimated * sample_rate),
    ) + window // 2
    if cut <= 0 or cut >= len(audio) - int(0.4 * sample_rate):
        return None
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(destination), audio[cut:], sample_rate, subtype="PCM_16")
    return {
        "prefix_chars": prefix_chars,
        "estimated_seconds": round(estimated, 3),
        "trimmed_seconds": round(cut / sample_rate, 3),
    }


def _normalize_alignment_text(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]", "", text)


def mandarin_to_wu_text(text: str) -> str:
    """Convert Mandarin output into a lightweight Shanghainese/Wu oral script.

    This is intentionally a transparent glossary rewrite rather than a hidden
    model. It gives the user an editable Wu-style script before speech synthesis.
    """

    converted = text
    for source, target in MANDARIN_TO_WU_RULES:
        converted = converted.replace(source, target)
    converted = converted.replace("的", "个")
    converted = converted.replace("吗？", "伐？")
    converted = converted.replace("吗", "伐")
    converted = converted.replace("了。", "哉。")
    converted = converted.replace("辰光辰光", "辰光")
    converted = converted.replace("电话号码号码", "电话号码")
    converted = converted.replace("求助电话号码号码", "求助电话号码")
    converted = converted.replace("请侬本拧", "请侬本人")
    converted = converted.replace("本拧", "本人")
    converted = converted.replace("别拧", "别人")
    converted = converted.replace("他拧", "他人")
    converted = converted.replace("只打12345", "只拨12345")
    return converted


def is_true_wu_voice(voice: str) -> bool:
    normalized = voice.lower()
    return normalized.startswith("wuu-") or "shanghai" in normalized or "wu-" in normalized


def is_voice_clone_backend(backend: str) -> bool:
    return backend.strip().lower() in {
        "cosyvoice_wu",
        "cosyvoice-wu",
        "gpt_sovits",
        "gpt-sovits",
        "sovits",
        "command",
        "external",
    }


def wu_voice_notice(voice: str = DEFAULT_WU_TTS_VOICE, backend: str = DEFAULT_WU_TTS_BACKEND) -> str | None:
    if is_voice_clone_backend(backend):
        return None
    if is_true_wu_voice(voice):
        return None
    return (
        "当前 edge-tts 没有可用的 wuu-CN/上海话音色；"
        "已按训练集风格生成吴语口语稿，但 MP3 仍会用普通话音色朗读。"
    )


def resolve_tts_text(
    target: str,
    mandarin_text: str,
    answer_text: str | None = None,
    *,
    backend: str | None = None,
) -> str:
    # A native Wu acoustic model already supplies dialect pronunciation. Feeding
    # it mechanically rewritten dialect characters moves the text outside the
    # model's normal training distribution and can cause inserted or blurred words.
    native_wu_backend = (backend or "").strip().lower() in {"cosyvoice_wu", "cosyvoice-wu"}
    if target == "wuu":
        if native_wu_backend:
            return mandarin_text
        return mandarin_to_wu_text(mandarin_text)
    if target == "answer":
        return answer_text or mandarin_text
    if target == "answer_wuu":
        if native_wu_backend:
            return answer_text or mandarin_text
        return mandarin_to_wu_text(answer_text or mandarin_text)
    return mandarin_text


def synthesize_mp3(request: TTSRequest) -> Path:
    if not request.text.strip():
        raise ValueError("TTS text is empty.")
    output_path = Path(request.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    backend = request.backend.strip().lower()
    if backend in {"edge", "edge_tts", "edge-tts"}:
        _run_async(_synthesize_edge_tts(request, output_path))
    elif backend in {"cosyvoice_wu", "cosyvoice-wu"}:
        _synthesize_cosyvoice_wu(request, output_path)
    elif backend in {"gpt_sovits", "gpt-sovits", "sovits"}:
        _synthesize_gpt_sovits(request, output_path)
    elif backend in {"command", "external"}:
        _synthesize_command(request, output_path)
    else:
        raise ValueError(f"Unsupported TTS backend: {request.backend}")
    return output_path


async def _synthesize_edge_tts(request: TTSRequest, output_path: Path) -> None:
    try:
        import edge_tts
    except ImportError as exc:
        raise RuntimeError(
            "需要先安装语音合成依赖：python -m pip install edge-tts"
        ) from exc

    communicate = edge_tts.Communicate(
        text=request.text,
        voice=request.voice,
        rate=request.rate,
        pitch=request.pitch,
    )
    await communicate.save(str(output_path))


def _synthesize_cosyvoice_wu(request: TTSRequest, output_path: Path) -> None:
    chunks = split_tts_sentences(request.text, normalize_numbers=False)
    if not chunks:
        raise RuntimeError("CosyVoice-Wu text became empty after normalization.")
    audio_chunks = [
        _request_cosyvoice_wu_audio_bytes(replace(request, text=chunk))
        for chunk in chunks
    ]
    audio_bytes = (
        audio_chunks[0]
        if len(audio_chunks) == 1
        else _join_wav_chunks(audio_chunks, pause_seconds=0.28)
    )
    if output_path.suffix.lower() == ".mp3":
        _write_wav_bytes_as_mp3(audio_bytes, output_path)
    else:
        output_path.write_bytes(audio_bytes)


def _request_cosyvoice_wu_audio_bytes(request: TTSRequest) -> bytes:
    payload = {
        "text": request.text,
        "prompt_text": request.prompt_text,
        "prompt_audio": str(Path(request.ref_audio_path).resolve()) if request.ref_audio_path else None,
        "speed": request.speed,
        "instruction": request.instruction,
        "use_text_frontend": request.use_text_frontend,
        "seed": request.seed,
    }
    http_request = Request(
        request.cosyvoice_wu_url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    try:
        with urlopen(http_request, timeout=300) as response:
            audio_bytes = response.read()
    except HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"WenetSpeech-Wu 生成专家调用失败。HTTP {exc.code}: {error_body}"
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            "WenetSpeech-Wu 生成专家调用失败。请先运行 "
            "scripts\\start_wenet_wu_expert.ps1，并检查 --cosyvoice-wu-url。"
        ) from exc
    if not audio_bytes:
        raise RuntimeError("WenetSpeech-Wu 生成专家返回了空音频。")
    return audio_bytes


def build_gpt_sovits_payload(request: TTSRequest, output_path: Path) -> dict:
    ref_audio_path = request.ref_audio_path or default_wu_reference_audio()
    if not ref_audio_path:
        raise ValueError("GPT-SoVITS backend requires ref_audio_path.")
    media_type = output_path.suffix.lower().lstrip(".") or "wav"
    if media_type == "mp3":
        # GPT-SoVITS api_v2 does not support MP3 directly in some releases.
        # Request WAV and transcode locally so the public TTS API still returns
        # the file type requested by the app/CLI.
        media_type = "wav"
    ref_audio = Path(ref_audio_path).resolve()
    return {
        "text": request.text,
        "text_lang": request.text_lang,
        "ref_audio_path": str(ref_audio),
        "prompt_text": request.prompt_text or default_wu_prompt_text(),
        "prompt_lang": request.prompt_lang,
        "media_type": media_type,
        "streaming_mode": False,
        "text_split_method": "cut0",
        "fragment_interval": DEFAULT_GPT_SOVITS_FRAGMENT_INTERVAL,
        "speed_factor": DEFAULT_GPT_SOVITS_SPEED,
    }


def default_wu_reference_audio() -> Path | None:
    candidate = Path(DEFAULT_WU_REF_AUDIO)
    return candidate if candidate.exists() else None


def default_wu_prompt_text() -> str:
    prompt_file = Path(DEFAULT_WU_REF_PROMPT_FILE)
    if prompt_file.exists():
        return prompt_file.read_text(encoding="utf-8").strip()
    return ""


def _synthesize_gpt_sovits(request: TTSRequest, output_path: Path) -> None:
    chunks = split_gpt_sovits_text(normalize_tts_text_for_clarity(request.text))
    if not chunks:
        raise RuntimeError("GPT-SoVITS text became empty after normalization.")
    audio_chunks = [
        _request_gpt_sovits_audio_bytes(replace(request, text=chunk), output_path)
        for chunk in chunks
    ]
    audio_bytes = (
        audio_chunks[0]
        if len(audio_chunks) == 1
        else _join_wav_chunks(audio_chunks, pause_seconds=DEFAULT_GPT_SOVITS_FRAGMENT_INTERVAL)
    )
    if output_path.suffix.lower() == ".mp3":
        _write_wav_bytes_as_mp3(audio_bytes, output_path)
    else:
        output_path.write_bytes(audio_bytes)


def _request_gpt_sovits_audio_bytes(request: TTSRequest, output_path: Path) -> bytes:
    payload = build_gpt_sovits_payload(request, output_path)
    http_request = Request(
        request.gpt_sovits_url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(http_request, timeout=180) as response:
            audio_bytes = response.read()
    except HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            "GPT-SoVITS 合成失败。"
            f"HTTP {exc.code}: {error_body}"
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            "GPT-SoVITS 合成失败。请确认 api_v2.py 已启动，并检查 --gpt-sovits-url、"
            "--ref-audio 和 --prompt-text。"
        ) from exc
    if not audio_bytes:
        raise RuntimeError("GPT-SoVITS returned empty audio.")
    return audio_bytes


def normalize_tts_text_for_clarity(text: str, *, convert_numbers: bool = True) -> str:
    """Make hotline-style Wu output easier for GPT-SoVITS to pronounce clearly."""

    digit_map = str.maketrans({
        "0": "零",
        "1": "一",
        "2": "二",
        "3": "三",
        "4": "四",
        "5": "五",
        "6": "六",
        "7": "七",
        "8": "八",
        "9": "九",
    })

    def convert_digits(match: re.Match[str]) -> str:
        return "，".join(match.group(0).translate(digit_map))

    normalized = re.sub(r"\d{2,}", convert_digits, text) if convert_numbers else text
    normalized = re.sub(r"\s+", "", normalized)
    normalized = normalized.replace("：", "。")
    normalized = normalized.replace(":", "。")
    normalized = normalized.replace("；", "。")
    normalized = normalized.replace(";", "。")
    normalized = normalized.replace("、", "，")
    normalized = normalized.replace("，", "，")
    normalized = re.sub(r"[。！？!?]+", "。", normalized)
    normalized = re.sub(r"，+", "，", normalized)
    return normalized.strip()


def split_gpt_sovits_text(text: str, max_chars: int = DEFAULT_GPT_SOVITS_MAX_CHARS) -> list[str]:
    chunks: list[str] = []
    buffer = ""
    for part in re.split(r"([。！？!?，,])", text):
        if not part:
            continue
        candidate = buffer + part
        if len(candidate) <= max_chars:
            buffer = candidate
            if part in "。！？!?":
                chunks.append(buffer.strip())
                buffer = ""
            continue
        if buffer.strip():
            chunks.extend(_hard_split_tts_chunk(buffer.strip(), max_chars))
        buffer = part
    if buffer.strip():
        chunks.extend(_hard_split_tts_chunk(buffer.strip(), max_chars))
    return [chunk for chunk in chunks if chunk.strip("，,。！？!? ")]


def split_tts_sentences(
    text: str,
    max_chars: int = 28,
    *,
    normalize_numbers: bool = True,
) -> list[str]:
    """Split an answer into independently generatable and verifiable clauses."""

    normalized = normalize_tts_text_for_clarity(
        text,
        convert_numbers=normalize_numbers,
    )
    clauses: list[str] = []
    buffer = ""
    for part in re.split(r"([。！？!?；;])", normalized):
        if not part:
            continue
        buffer += part
        if part in "。！？!?；;":
            clauses.extend(_split_semantic_clause(buffer, max_chars))
            buffer = ""
    if buffer.strip():
        clauses.extend(_split_semantic_clause(buffer, max_chars))
    return [clause for clause in clauses if clause.strip("，,。！？!?；; ")]


def _split_semantic_clause(text: str, max_chars: int) -> list[str]:
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    # Keep digit-by-digit phone numbers atomic while splitting at commas.
    # Otherwise `一，二，三，四，五` can become an unsafe `1234` + `5` pair.
    digit_chars = "零一二三四五六七八九"
    protected = re.sub(
        rf"(?<=[{digit_chars}])[，,](?=[{digit_chars}])",
        "\x00",
        text,
    )
    pieces: list[str] = []
    current = ""
    for token in re.split(r"([，,])", protected):
        if not token:
            continue
        if current and len(current) + len(token) > max_chars:
            pieces.extend(_hard_split_tts_chunk(current.strip(), max_chars))
            current = token.lstrip("，,")
        else:
            current += token
    if current.strip():
        pieces.extend(_hard_split_tts_chunk(current.strip(), max_chars))
    return [piece.replace("\x00", "，") for piece in pieces]


def join_wav_files(
    input_paths: list[str | Path],
    output_path: str | Path,
    pause_seconds: float = 0.28,
) -> Path:
    """Join selected sentence WAV files and optionally transcode to MP3."""

    paths = [Path(path) for path in input_paths]
    if not paths:
        raise ValueError("No sentence audio files to join.")
    audio_bytes = _join_wav_chunks(
        [path.read_bytes() for path in paths],
        pause_seconds=pause_seconds,
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.suffix.lower() == ".mp3":
        _write_wav_bytes_as_mp3(audio_bytes, destination)
    else:
        destination.write_bytes(audio_bytes)
    return destination


def _hard_split_tts_chunk(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    pieces: list[str] = []
    current = ""
    for token in re.split(r"([，,])", text):
        if not token:
            continue
        if len(current) + len(token) > max_chars and current:
            pieces.append(current.strip())
            current = token.lstrip("，,")
        else:
            current += token
    if current.strip():
        pieces.append(current.strip())
    final: list[str] = []
    for piece in pieces:
        if len(piece) <= max_chars:
            final.append(piece)
        else:
            final.extend(piece[index : index + max_chars] for index in range(0, len(piece), max_chars))
    return final


def _join_wav_chunks(audio_chunks: list[bytes], pause_seconds: float = 0.3) -> bytes:
    first_params = None
    frames: list[bytes] = []
    for audio in audio_chunks:
        with wave.open(io.BytesIO(audio), "rb") as reader:
            params = reader.getparams()
            if first_params is None:
                first_params = params
            elif params[:3] != first_params[:3]:
                raise RuntimeError("GPT-SoVITS returned WAV chunks with incompatible formats.")
            frames.append(reader.readframes(reader.getnframes()))
    if first_params is None:
        raise RuntimeError("No GPT-SoVITS audio chunks to join.")
    silence_frames = int(first_params.framerate * pause_seconds)
    silence = b"\x00" * silence_frames * first_params.nchannels * first_params.sampwidth
    output = io.BytesIO()
    with wave.open(output, "wb") as writer:
        writer.setparams(first_params)
        for index, frame in enumerate(frames):
            if index:
                writer.writeframes(silence)
            writer.writeframes(frame)
    return output.getvalue()


def _write_wav_bytes_as_mp3(audio_bytes: bytes, output_path: Path) -> None:
    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError(
            "GPT-SoVITS returned WAV audio, but MP3 output requires ffmpeg. "
            "Install ffmpeg or use a .wav output path."
        )
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as handle:
        handle.write(audio_bytes)
        wav_path = Path(handle.name)
    try:
        completed = subprocess.run(
            [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(wav_path),
                "-codec:a",
                "libmp3lame",
                "-q:a",
                "3",
                str(output_path),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
    finally:
        wav_path.unlink(missing_ok=True)
    if completed.returncode != 0:
        raise RuntimeError(f"ffmpeg MP3 conversion failed: {completed.stderr}")
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError("ffmpeg did not create a non-empty MP3 file.")


def _find_ffmpeg() -> str | None:
    configured = os.environ.get("FFMPEG_BINARY") or os.environ.get("IMAGEIO_FFMPEG_EXE")
    if configured and Path(configured).exists():
        return configured
    bundled = Path("external/ffmpeg-shared/bin/ffmpeg.exe")
    if bundled.exists():
        return str(bundled)
    return shutil.which("ffmpeg")


def _synthesize_command(request: TTSRequest, output_path: Path) -> None:
    template = request.command_template or os.environ.get("SHANGHAI_TTS_COMMAND")
    if not template:
        raise ValueError("Command TTS backend requires --tts-command or SHANGHAI_TTS_COMMAND.")
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as handle:
        handle.write(request.text)
        text_file = Path(handle.name)
    try:
        command = template.format(
            text=shlex.quote(request.text),
            text_file=str(text_file),
            output=str(output_path),
            ref_audio=str(request.ref_audio_path or ""),
            prompt_text=shlex.quote(request.prompt_text),
            text_lang=request.text_lang,
            prompt_lang=request.prompt_lang,
        )
        completed = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=600)
    finally:
        text_file.unlink(missing_ok=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "External TTS command failed.\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError("External TTS command did not create a non-empty output file.")


def _run_async(coro) -> None:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(coro)
        return

    error: list[BaseException] = []

    def runner() -> None:
        try:
            asyncio.run(coro)
        except BaseException as exc:
            error.append(exc)

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()
    if error:
        raise error[0]
