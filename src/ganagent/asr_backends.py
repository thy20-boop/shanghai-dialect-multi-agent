from __future__ import annotations

from abc import ABC, abstractmethod
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Callable, Iterable

from ganagent.models import Segment


KNOWN_PEFT_ADAPTER_MODELS = {"kaiwang0574/whisper-wu"}
CONTROL_TOKEN_RE = re.compile(r"<\|[^|>]+?\|>|<[A-Za-z_][A-Za-z0-9_-]*>")
DEFAULT_DOLPHIN_HOTWORDS = [
    "先生",
    "侬好",
    "初次见面",
    "第一趟",
    "上海",
    "上海话",
    "是额",
    "老早",
    "阿拉",
    "拧",
    "拧来",
    "搿搭",
    "辰光",
    "啥",
    "勿",
    "伐",
]
DEFAULT_TRAINED_WHISPER_LORA = Path("outputs/models/whisper-small-shanghai-lora-full")
DEFAULT_WHISPER_BASE_MODEL = "TingChen-ppmc/whisper-small-Shanghai"


def default_wu_runtime_root() -> Path:
    configured = os.environ.get("SHANGHAI_WU_RUNTIME")
    if configured:
        return Path(configured)
    legacy = Path(r"D:\wswu_runtime")
    if legacy.exists():
        return legacy
    local_app_data = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    return local_app_data / "ShanghaiDialectAgent" / "wswu_runtime"


DEFAULT_WU_RUNTIME_ROOT = default_wu_runtime_root()
DEFAULT_WHISPER_MEDIUM_WU_MODEL = Path(
    os.environ.get(
        "SHANGHAI_WHISPER_MEDIUM_WU_MODEL",
        str(DEFAULT_WU_RUNTIME_ROOT / "models" / "Whisper-Medium-Wu" / "whisper"),
    )
)
DEFAULT_WENET_SOURCE = Path(
    os.environ.get("SHANGHAI_WENET_SOURCE", str(DEFAULT_WU_RUNTIME_ROOT / "wenet-main"))
)


def default_trained_whisper_model() -> str:
    return str(DEFAULT_TRAINED_WHISPER_LORA) if DEFAULT_TRAINED_WHISPER_LORA.exists() else DEFAULT_WHISPER_BASE_MODEL


def normalize_model_reference(model_name: str) -> tuple[str, bool]:
    """Return a loadable model id and whether it should be treated as PEFT."""

    if model_name.startswith("peft:"):
        return model_name.removeprefix("peft:"), True
    return model_name, model_name in KNOWN_PEFT_ADAPTER_MODELS


def strip_asr_control_tokens(text: str) -> str:
    return CONTROL_TOKEN_RE.sub("", text).strip()


def parse_hotwords(raw: str | None, defaults: list[str] | None = None) -> list[str]:
    if raw is None:
        return list(defaults or [])
    hotwords = [
        item.strip()
        for item in re.split(r"[,，;；\n\r\t ]+", raw)
        if item.strip()
    ]
    return hotwords


class ASRBackend(ABC):
    name = "base"

    @abstractmethod
    def transcribe(self, audio_path: str | None = None) -> list[Segment]:
        raise NotImplementedError


class MockASRBackend(ASRBackend):
    """A deterministic backend for demos and tests."""

    name = "mock"

    def __init__(self, segments: Iterable[Segment] | None = None) -> None:
        self._segments = list(segments) if segments else None

    def transcribe(self, audio_path: str | None = None) -> list[Segment]:
        if self._segments is not None:
            return self._segments
        return [
            Segment(
                start=0.0,
                end=3.2,
                text="侬好伐？今朝搿个模型识别得交关灵。",
                confidence=0.86,
                backend=self.name,
                language_hint="wuu",
            ),
            Segment(
                start=3.2,
                end=8.6,
                text="阿拉个大语言摸型课要做罗拉微信，还要评估塞尔。",
                confidence=0.61,
                backend=self.name,
                language_hint="wuu",
            ),
            Segment(
                start=8.6,
                end=12.0,
                text="伊讲个词吾听勿大清爽，等歇要再校一遍。",
                confidence=0.72,
                backend=self.name,
                language_hint="wuu",
            ),
        ]


class WhisperTransformersBackend(ASRBackend):
    """Optional Transformers backend.

    The import is lazy so the core agent remains runnable without heavy ASR
    dependencies.
    """

    name = "whisper"

    def __init__(
        self,
        model_name: str = "openai/whisper-small",
        device: str | None = None,
        local_files_only: bool = False,
        chunk_seconds: float = 15.0,
        vad_enabled: bool | None = None,
        max_speech_region_seconds: float | None = None,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.local_files_only = local_files_only
        self.chunk_seconds = chunk_seconds
        self.vad_enabled = (
            os.environ.get("SHANGHAI_ASR_USE_VAD", "1") != "0"
            if vad_enabled is None
            else vad_enabled
        )
        self.max_speech_region_seconds = float(
            max_speech_region_seconds
            if max_speech_region_seconds is not None
            else os.environ.get("SHANGHAI_ASR_MAX_REGION_SECONDS", "8.0")
        )
        self.progress_callback = progress_callback
        self._pipe = None

    def set_progress_callback(
        self,
        callback: Callable[[float, str], None] | None,
    ) -> None:
        self.progress_callback = callback

    def transcribe(self, audio_path: str | None = None) -> list[Segment]:
        if not audio_path:
            raise ValueError("Whisper backend requires --audio.")
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(path)

        self._notify(0.02, "正在加载上海话模型...")
        pipe = self._get_pipeline()
        self._notify(0.12, "模型已加载，正在读取音频...")
        audio_input = load_audio_for_pipeline(path, pipe.feature_extractor.sampling_rate)
        if isinstance(audio_input, dict) and self.chunk_seconds:
            duration = len(audio_input["raw"]) / audio_input["sampling_rate"]
            if duration > self._effective_chunk_seconds():
                return self._transcribe_chunked(pipe, audio_input)

        self._notify(0.2, "正在识别音频...")
        output = pipe(audio_input)
        self._notify(0.95, "识别完成，正在整理结果...")
        chunks = output.get("chunks") or []
        if not chunks:
            self._notify(1.0, "识别完成")
            return [
                Segment(
                    start=0.0,
                    end=0.0,
                    text=output.get("text", "").strip(),
                    confidence=None,
                    backend=self.name,
                    language_hint="zh",
                )
            ]

        segments: list[Segment] = []
        for chunk in chunks:
            start, end = chunk.get("timestamp") or (0.0, 0.0)
            segments.append(
                Segment(
                    start=float(start or 0.0),
                    end=float(end or 0.0),
                    text=(chunk.get("text") or "").strip(),
                    confidence=None,
                    backend=self.name,
                    language_hint="zh",
                )
            )
        self._notify(1.0, "识别完成")
        return segments

    def _get_pipeline(self):
        if self._pipe is not None:
            return self._pipe

        try:
            from huggingface_hub.utils import disable_progress_bars
            import torch
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
            from transformers import pipeline
            from transformers.utils import logging as hf_logging
        except ImportError as exc:
            raise RuntimeError(
                "Install ASR extras first: python -m pip install -e .[asr]"
            ) from exc

        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        disable_progress_bars()
        hf_logging.disable_progress_bar()
        hf_logging.set_verbosity_error()
        device = self.device
        if device is None:
            device = 0 if torch.cuda.is_available() else -1
        model_name_or_path, force_peft = normalize_model_reference(self.model_name)
        adapter_path = Path(model_name_or_path)
        is_lora_adapter = adapter_path.exists() and (adapter_path / "adapter_config.json").exists()
        processor_name_or_path = model_name_or_path
        if is_lora_adapter or force_peft:
            try:
                from peft import PeftConfig, PeftModel
            except ImportError as exc:
                raise RuntimeError(
                    "Install PEFT first to load a LoRA adapter: python -m pip install peft"
                ) from exc
            peft_config = PeftConfig.from_pretrained(
                model_name_or_path,
                local_files_only=self.local_files_only,
            )
            base_model_name = peft_config.base_model_name_or_path or "TingChen-ppmc/whisper-small-Shanghai"
            bundled_base_model = Path("models/whisper-small-Shanghai")
            if base_model_name == "TingChen-ppmc/whisper-small-Shanghai" and bundled_base_model.exists():
                base_model_name = str(bundled_base_model)
            processor_name_or_path = (
                model_name_or_path
                if (adapter_path / "preprocessor_config.json").exists()
                else base_model_name
            )
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                base_model_name,
                local_files_only=self.local_files_only,
            )
            model = PeftModel.from_pretrained(model, model_name_or_path, is_trainable=False)
            if hasattr(model, "merge_and_unload"):
                model = model.merge_and_unload()
        else:
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_name_or_path,
                local_files_only=self.local_files_only,
            )
        processor = AutoProcessor.from_pretrained(
            processor_name_or_path,
            local_files_only=self.local_files_only,
        )
        self._pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device=device,
            generate_kwargs={
                "language": "Chinese",
                "task": "transcribe",
                "num_beams": int(os.environ.get("SHANGHAI_ASR_NUM_BEAMS", "5")),
                "no_repeat_ngram_size": int(os.environ.get("SHANGHAI_ASR_NO_REPEAT_NGRAM_SIZE", "6")),
                "condition_on_prev_tokens": False,
                "temperature": float(os.environ.get("SHANGHAI_ASR_TEMPERATURE", "0.0")),
                "compression_ratio_threshold": float(os.environ.get("SHANGHAI_ASR_COMPRESSION_RATIO_THRESHOLD", "2.4")),
                "logprob_threshold": float(os.environ.get("SHANGHAI_ASR_LOGPROB_THRESHOLD", "-1.0")),
                "no_speech_threshold": float(os.environ.get("SHANGHAI_ASR_NO_SPEECH_THRESHOLD", "0.6")),
            },
        )
        return self._pipe

    def _transcribe_chunked(self, pipe, audio_input: dict) -> list[Segment]:
        import numpy as np

        raw = audio_input["raw"]
        sampling_rate = int(audio_input["sampling_rate"])
        regions = self._speech_regions(raw, sampling_rate) if self.vad_enabled else []
        if regions:
            self._notify(0.14, f"已按停顿切成 {len(regions)} 个语音片段")
        else:
            chunk_samples = max(1, int(self._effective_chunk_seconds() * sampling_rate))
            regions = [
                (start_sample, min(len(raw), start_sample + chunk_samples))
                for start_sample in range(0, len(raw), chunk_samples)
            ]
            self._notify(0.14, f"未检测到稳定停顿，改用 {len(regions)} 个固定片段")

        total_chunks = len(regions)
        segments: list[Segment] = []

        for chunk_index, (start_sample, end_sample) in enumerate(regions):
            chunk = raw[start_sample:end_sample]
            self._notify(
                0.15 + 0.8 * chunk_index / total_chunks,
                f"正在识别第 {chunk_index + 1}/{total_chunks} 段...",
            )
            if len(chunk) < sampling_rate * 1.0:
                self._notify(
                    0.15 + 0.8 * (chunk_index + 1) / total_chunks,
                    f"已完成第 {chunk_index + 1}/{total_chunks} 段",
                )
                continue
            if float(np.sqrt(np.mean(np.square(chunk)))) < 0.002:
                self._notify(
                    0.15 + 0.8 * (chunk_index + 1) / total_chunks,
                    f"已跳过静音片段 {chunk_index + 1}/{total_chunks}",
                )
                continue

            chunk = self._normalize_chunk_for_model(chunk)
            output = pipe({"raw": chunk, "sampling_rate": sampling_rate})
            text = (output.get("text") or "").strip()
            self._notify(
                0.15 + 0.8 * (chunk_index + 1) / total_chunks,
                f"已完成第 {chunk_index + 1}/{total_chunks} 段",
            )
            if not text:
                continue

            segments.append(
                Segment(
                    start=start_sample / sampling_rate,
                    end=end_sample / sampling_rate,
                    text=text,
                    confidence=None,
                    backend=f"{self.name}_vad" if self.vad_enabled else self.name,
                    language_hint="zh",
                )
            )

        if segments:
            self._notify(1.0, "识别完成")
            return segments
        self._notify(1.0, "识别完成，但没有检测到语音")
        return [
            Segment(
                start=0.0,
                end=0.0,
                text="",
                confidence=None,
                backend=self.name,
                language_hint="zh",
            )
        ]

    def _effective_chunk_seconds(self) -> float:
        # Whisper switches to long-form mode above 30 seconds and then requires
        # timestamp generation. Keep manual chunks safely below that boundary.
        return min(max(float(self.chunk_seconds), 1.0), 29.0)

    def _max_region_seconds(self) -> float:
        return min(self._effective_chunk_seconds(), max(1.5, self.max_speech_region_seconds))

    def _speech_regions(self, raw, sampling_rate: int) -> list[tuple[int, int]]:
        import numpy as np

        audio = np.asarray(raw, dtype=np.float32)
        if audio.size < sampling_rate:
            return [(0, int(audio.size))] if audio.size else []

        audio = np.nan_to_num(audio)
        frame_samples = max(1, int(0.03 * sampling_rate))
        hop_samples = max(1, int(0.01 * sampling_rate))
        if audio.size <= frame_samples:
            rms = np.array([float(np.sqrt(np.mean(np.square(audio))))], dtype=np.float32)
            starts = np.array([0], dtype=np.int64)
        else:
            frame_count = 1 + math.floor((audio.size - frame_samples) / hop_samples)
            starts = np.arange(frame_count, dtype=np.int64) * hop_samples
            rms = np.empty(frame_count, dtype=np.float32)
            for index, start in enumerate(starts):
                frame = audio[start : start + frame_samples]
                rms[index] = float(np.sqrt(np.mean(np.square(frame))))

        if rms.size == 0 or float(np.max(rms)) < 0.001:
            return []

        p20, p50, p95 = np.percentile(rms, [20, 50, 95])
        threshold = max(0.0015, float(p20) * 3.0, float(p50) * 1.8, float(p95) * 0.10)
        speech = rms >= threshold
        speech = self._fill_short_gaps(speech, max_gap_frames=max(1, int(0.28 / 0.01)))
        speech = self._remove_short_regions(speech, min_region_frames=max(1, int(0.22 / 0.01)))
        if not bool(np.any(speech)):
            return []

        padding = int(0.28 * sampling_rate)
        raw_regions: list[tuple[int, int]] = []
        index = 0
        while index < speech.size:
            if not speech[index]:
                index += 1
                continue
            start_index = index
            while index < speech.size and speech[index]:
                index += 1
            end_index = index - 1
            start_sample = max(0, int(starts[start_index]) - padding)
            end_sample = min(audio.size, int(starts[end_index]) + frame_samples + padding)
            if end_sample - start_sample >= int(0.35 * sampling_rate):
                raw_regions.append((start_sample, end_sample))

        merged = self._merge_regions(
            raw_regions,
            sampling_rate,
            max_gap_seconds=0.35,
            max_region_seconds=self._max_region_seconds(),
        )
        merged = self._merge_short_regions(
            merged,
            sampling_rate,
            max_gap_seconds=0.9,
            short_region_seconds=1.8,
            max_region_seconds=self._max_region_seconds(),
        )
        limited: list[tuple[int, int]] = []
        max_samples = int(self._max_region_seconds() * sampling_rate)
        for start, end in merged:
            limited.extend(self._limit_region_length(audio, start, end, sampling_rate, max_samples))
        return limited

    @staticmethod
    def _fill_short_gaps(mask, max_gap_frames: int):
        import numpy as np

        filled = np.asarray(mask, dtype=bool).copy()
        index = 0
        while index < filled.size:
            if filled[index]:
                index += 1
                continue
            start = index
            while index < filled.size and not filled[index]:
                index += 1
            end = index
            if start > 0 and end < filled.size and end - start <= max_gap_frames:
                filled[start:end] = True
        return filled

    @staticmethod
    def _remove_short_regions(mask, min_region_frames: int):
        import numpy as np

        cleaned = np.asarray(mask, dtype=bool).copy()
        index = 0
        while index < cleaned.size:
            if not cleaned[index]:
                index += 1
                continue
            start = index
            while index < cleaned.size and cleaned[index]:
                index += 1
            end = index
            if end - start < min_region_frames:
                cleaned[start:end] = False
        return cleaned

    @staticmethod
    def _merge_regions(
        regions: list[tuple[int, int]],
        sampling_rate: int,
        max_gap_seconds: float,
        max_region_seconds: float,
    ) -> list[tuple[int, int]]:
        if not regions:
            return []
        max_gap = int(max_gap_seconds * sampling_rate)
        max_len = int(max_region_seconds * sampling_rate)
        merged: list[tuple[int, int]] = [regions[0]]
        for start, end in regions[1:]:
            prev_start, prev_end = merged[-1]
            if start - prev_end <= max_gap and end - prev_start <= max_len:
                merged[-1] = (prev_start, end)
            else:
                merged.append((start, end))
        return merged

    @staticmethod
    def _merge_short_regions(
        regions: list[tuple[int, int]],
        sampling_rate: int,
        max_gap_seconds: float,
        short_region_seconds: float,
        max_region_seconds: float,
    ) -> list[tuple[int, int]]:
        if not regions:
            return []
        max_gap = int(max_gap_seconds * sampling_rate)
        short_len = int(short_region_seconds * sampling_rate)
        max_len = int(max_region_seconds * sampling_rate)
        merged: list[tuple[int, int]] = [regions[0]]
        for start, end in regions[1:]:
            prev_start, prev_end = merged[-1]
            prev_short = prev_end - prev_start <= short_len
            current_short = end - start <= short_len
            if (prev_short or current_short) and start - prev_end <= max_gap and end - prev_start <= max_len:
                merged[-1] = (prev_start, end)
            else:
                merged.append((start, end))
        return merged

    def _limit_region_length(
        self,
        audio,
        start: int,
        end: int,
        sampling_rate: int,
        max_samples: int,
    ) -> list[tuple[int, int]]:
        if end - start <= max_samples:
            return [(start, end)]

        regions: list[tuple[int, int]] = []
        cursor = start
        min_samples = int(1.0 * sampling_rate)
        while end - cursor > max_samples:
            target = cursor + max_samples
            search_start = max(cursor + min_samples, target - int(0.9 * sampling_rate))
            search_end = min(end - min_samples, target + int(0.9 * sampling_rate))
            cut = self._lowest_energy_cut(audio, search_start, search_end, target, sampling_rate)
            regions.append((cursor, cut))
            cursor = cut
        if end - cursor >= int(0.35 * sampling_rate):
            regions.append((cursor, end))
        return regions

    @staticmethod
    def _lowest_energy_cut(audio, search_start: int, search_end: int, fallback: int, sampling_rate: int) -> int:
        import numpy as np

        if search_end <= search_start:
            return fallback
        frame = max(1, int(0.18 * sampling_rate))
        hop = max(1, int(0.04 * sampling_rate))
        best_cut = fallback
        best_energy = float("inf")
        for start in range(search_start, max(search_start + 1, search_end - frame), hop):
            window = audio[start : start + frame]
            if window.size == 0:
                continue
            energy = float(np.sqrt(np.mean(np.square(window))))
            if energy < best_energy:
                best_energy = energy
                best_cut = start + frame // 2
        return best_cut

    @staticmethod
    def _normalize_chunk_for_model(chunk):
        import numpy as np

        audio = np.asarray(chunk, dtype=np.float32).copy()
        if audio.size == 0:
            return audio
        audio = np.nan_to_num(audio)
        audio = audio - float(np.mean(audio))
        rms = float(np.sqrt(np.mean(np.square(audio))))
        if rms > 0.0:
            target_rms = float(os.environ.get("SHANGHAI_ASR_TARGET_RMS", "0.065"))
            gain = min(3.0, target_rms / rms) if rms < target_rms else 1.0
            audio *= gain
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 0.98:
            audio *= 0.98 / peak
        return audio.astype(np.float32, copy=False)

    def _notify(self, progress: float, message: str) -> None:
        if self.progress_callback is None:
            return
        try:
            self.progress_callback(min(max(progress, 0.0), 1.0), message)
        except Exception:
            # UI progress must never interrupt transcription.
            pass


class FunASRBackend(ASRBackend):
    """Optional FunASR/SenseVoice backend for open-source model comparison."""

    name = "funasr"

    def __init__(
        self,
        model_name: str = "iic/SenseVoiceSmall",
        device: str | None = None,
        local_files_only: bool = False,
        max_speech_region_seconds: float | None = None,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.local_files_only = local_files_only
        self.max_speech_region_seconds = float(max_speech_region_seconds or 30.0)
        self.progress_callback = progress_callback
        self._model = None

    def set_progress_callback(
        self,
        callback: Callable[[float, str], None] | None,
    ) -> None:
        self.progress_callback = callback

    def transcribe(self, audio_path: str | None = None) -> list[Segment]:
        if not audio_path:
            raise ValueError("FunASR backend requires --audio.")
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(path)

        self._notify(0.02, "正在加载开源 FunASR 模型...")
        model = self._get_model()
        self._notify(0.12, "模型已加载，正在转换音频...")
        with normalized_temp_wav(path, 16000) as wav_path:
            self._notify(0.2, "正在使用 FunASR/SenseVoice 识别...")
            result = model.generate(
                input=str(wav_path),
                cache={},
                language=os.environ.get("SHANGHAI_FUNASR_LANGUAGE", "auto"),
                use_itn=True,
                batch_size_s=int(os.environ.get("SHANGHAI_FUNASR_BATCH_SECONDS", "60")),
                merge_vad=True,
                merge_length_s=int(os.environ.get("SHANGHAI_FUNASR_MERGE_SECONDS", "15")),
            )
        self._notify(1.0, "识别完成")
        return self._segments_from_result(result)

    def _get_model(self):
        if self._model is not None:
            return self._model
        try:
            import torch
            from funasr import AutoModel
        except ImportError as exc:
            raise RuntimeError(
                "需要先安装开源候选后端：python -m pip install funasr"
            ) from exc

        device = self.device
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        vad_ms = int(max(1.0, self.max_speech_region_seconds) * 1000)
        kwargs = {
            "model": self.model_name or "iic/SenseVoiceSmall",
            "trust_remote_code": True,
            "vad_model": "fsmn-vad",
            "vad_kwargs": {"max_single_segment_time": vad_ms},
            "device": device,
        }
        if str(kwargs["model"]).startswith("FunAudioLLM/"):
            kwargs["hub"] = "hf"
        self._model = AutoModel(**kwargs)
        return self._model

    def _segments_from_result(self, result) -> list[Segment]:
        rows = result if isinstance(result, list) else [result]
        segments: list[Segment] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            sentence_info = row.get("sentence_info") or []
            for sent in sentence_info:
                text = strip_asr_control_tokens(str(sent.get("text") or ""))
                if not text:
                    continue
                segments.append(
                    Segment(
                        start=float(sent.get("start") or 0.0) / 1000.0,
                        end=float(sent.get("end") or 0.0) / 1000.0,
                        text=text,
                        confidence=None,
                        backend=self.name,
                        language_hint="zh",
                    )
                )
            if not segments:
                text = strip_asr_control_tokens(str(row.get("text") or ""))
                if text:
                    segments.append(
                        Segment(
                            start=0.0,
                            end=0.0,
                            text=text,
                            confidence=None,
                            backend=self.name,
                            language_hint="zh",
                        )
                    )
        return segments or [
            Segment(start=0.0, end=0.0, text="", confidence=None, backend=self.name, language_hint="zh")
        ]

    def _notify(self, progress: float, message: str) -> None:
        if self.progress_callback is None:
            return
        try:
            self.progress_callback(min(max(progress, 0.0), 1.0), message)
        except Exception:
            pass


class WhisperMediumWuBackend(ASRBackend):
    """Official WenetSpeech-Wu Whisper-Medium expert running on WeNet."""

    name = "whisper_medium_wu"

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        local_files_only: bool = False,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> None:
        self.model_dir = Path(model_name) if model_name else DEFAULT_WHISPER_MEDIUM_WU_MODEL
        self.device = device
        self.local_files_only = local_files_only
        self.progress_callback = progress_callback
        self._model = None

    def set_progress_callback(
        self,
        callback: Callable[[float, str], None] | None,
    ) -> None:
        self.progress_callback = callback

    def transcribe(self, audio_path: str | None = None) -> list[Segment]:
        if not audio_path:
            raise ValueError("Whisper-Medium-Wu backend requires --audio.")
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(path)

        self._notify(0.02, "正在加载 Whisper-Medium-Wu 专家模型...")
        model, torch = self._get_model()
        self._notify(0.2, "Whisper-Medium-Wu 正在复核吴语音频...")
        with normalized_temp_wav(path, 16000) as wav_path:
            chunk_paths = self._chunk_long_wav(wav_path)
            try:
                texts: list[str] = []
                for index, chunk_path in enumerate(chunk_paths, start=1):
                    self._notify(
                        0.2 + 0.75 * index / len(chunk_paths),
                        f"Whisper-Medium-Wu 正在复核第 {index}/{len(chunk_paths)} 段...",
                    )
                    if next(model.parameters()).device.type == "cuda":
                        with torch.inference_mode(), torch.autocast(
                            device_type="cuda",
                            dtype=torch.float16,
                        ):
                            result = model.transcribe(str(chunk_path))
                    else:
                        with torch.inference_mode():
                            result = model.transcribe(str(chunk_path))
                    texts.append(
                        strip_asr_control_tokens(
                            str(getattr(result, "text", result) or "")
                        )
                    )
            finally:
                for chunk_path in chunk_paths:
                    if chunk_path != wav_path:
                        chunk_path.unlink(missing_ok=True)
        text = "".join(texts)
        self._notify(1.0, "Whisper-Medium-Wu 复核完成")
        return [
            Segment(
                start=0.0,
                end=0.0,
                text=text,
                confidence=None,
                backend=self.name,
                language_hint="wuu",
            )
        ]

    @staticmethod
    def _chunk_long_wav(wav_path: Path, max_seconds: float = 24.0) -> list[Path]:
        """Split long input near the quietest boundary below Whisper's limit."""

        import numpy as np
        import soundfile as sf

        audio, sample_rate = sf.read(str(wav_path), dtype="float32")
        if getattr(audio, "ndim", 1) > 1:
            audio = audio.mean(axis=1)
        max_samples = int(max_seconds * sample_rate)
        if len(audio) <= max_samples:
            return [wav_path]

        paths: list[Path] = []
        start = 0
        while start < len(audio):
            target = min(start + max_samples, len(audio))
            end = target
            if target < len(audio):
                search_start = max(start + int(8 * sample_rate), target - int(3 * sample_rate))
                search_end = min(len(audio), target + int(sample_rate))
                window = max(1, int(0.16 * sample_rate))
                hop = max(1, int(0.04 * sample_rate))
                candidates = range(search_start, max(search_start + 1, search_end - window), hop)
                end = min(
                    candidates,
                    key=lambda offset: float(np.mean(np.abs(audio[offset : offset + window]))),
                    default=target,
                ) + window // 2
            handle = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            handle.close()
            chunk_path = Path(handle.name)
            sf.write(str(chunk_path), audio[start:end], sample_rate, subtype="PCM_16")
            paths.append(chunk_path)
            start = end
        return paths

    def _get_model(self):
        if self._model is not None:
            return self._model
        config_path = self.model_dir / "train.yaml"
        checkpoint_path = self.model_dir / "whisper-medium.pt"
        if not config_path.exists() or not checkpoint_path.exists():
            raise RuntimeError(
                "Whisper-Medium-Wu 模型尚未安装。请运行 "
                "scripts\\setup_whisper_medium_wu.ps1。"
            )
        if not DEFAULT_WENET_SOURCE.exists():
            raise RuntimeError(
                "WeNet 运行框架尚未安装。请运行 "
                "scripts\\setup_whisper_medium_wu.ps1。"
            )
        source_path = str(DEFAULT_WENET_SOURCE)
        if source_path not in sys.path:
            sys.path.insert(0, source_path)
        try:
            import argparse
            import copy

            import torch
            import yaml
            from wenet.cli.model import load_feature
            from wenet.utils.init_model import init_model
            from wenet.utils.init_tokenizer import init_tokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Whisper-Medium-Wu 依赖不完整。请运行 "
                "scripts\\setup_whisper_medium_wu.ps1。"
            ) from exc

        configs = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        inference_configs = copy.deepcopy(configs)
        tokenizer = init_tokenizer(inference_configs)
        inference_configs.setdefault("output_dim", len(tokenizer.symbol_table))
        args = argparse.Namespace(checkpoint=str(checkpoint_path), jit=False)
        model, inference_configs = init_model(args, inference_configs)
        model.tokenizer = tokenizer
        model.compute_feature = load_feature(str(self.model_dir))[0]
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        if device == "cuda":
            model = model.half().to(device).eval()
        else:
            model = model.to(device).eval()
        self._model = (model, torch)
        return self._model

    def _notify(self, progress: float, message: str) -> None:
        if self.progress_callback is None:
            return
        try:
            self.progress_callback(min(max(progress, 0.0), 1.0), message)
        except Exception:
            pass


class DolphinBackend(ASRBackend):
    """Optional DataoceanAI Dolphin Chinese dialect backend."""

    name = "dolphin"

    def __init__(
        self,
        model_name: str = "small.cn",
        device: str | None = None,
        local_files_only: bool = False,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.local_files_only = local_files_only
        self.progress_callback = progress_callback
        self._model = None

    def set_progress_callback(
        self,
        callback: Callable[[float, str], None] | None,
    ) -> None:
        self.progress_callback = callback

    def transcribe(self, audio_path: str | None = None) -> list[Segment]:
        if not audio_path:
            raise ValueError("Dolphin backend requires --audio.")
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(path)

        self._notify(0.02, "正在加载开源 Dolphin 方言模型...")
        model, transcribe = self._get_model()
        self._notify(0.12, "模型已加载，正在转换音频...")
        with normalized_temp_wav(path, 16000) as wav_path:
            self._notify(0.2, "正在使用 Dolphin 识别...")
            result = transcribe(
                model,
                str(wav_path),
                lang_sym=os.environ.get("SHANGHAI_DOLPHIN_LANG", "zh"),
                region_sym=os.environ.get("SHANGHAI_DOLPHIN_REGION", "SHANGHAI"),
                hotwords=parse_hotwords(
                    os.environ.get("SHANGHAI_DOLPHIN_HOTWORDS"),
                    DEFAULT_DOLPHIN_HOTWORDS,
                ),
                use_deep_biasing=os.environ.get("SHANGHAI_DOLPHIN_DEEP_BIASING", "1") != "0",
                use_prompt_hotword=os.environ.get("SHANGHAI_DOLPHIN_PROMPT_HOTWORD", "0") != "0",
                use_two_stage_filter=os.environ.get("SHANGHAI_DOLPHIN_TWO_STAGE_FILTER", "1") != "0",
            )
        text = self._text_from_result(result)
        self._notify(1.0, "识别完成")
        return [
            Segment(
                start=0.0,
                end=0.0,
                text=text,
                confidence=None,
                backend=self.name,
                language_hint="zh",
            )
        ]

    def _get_model(self):
        if self._model is not None:
            return self._model
        try:
            import torch
            import dolphin
            from dolphin import transcribe
        except ImportError as exc:
            raise RuntimeError(
                "需要先安装开源候选后端：python -m pip install dataoceanai-dolphin"
            ) from exc
        device = self.device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = (dolphin.load_model(self.model_name or "small.cn", device=device), transcribe)
        return self._model

    @staticmethod
    def _text_from_result(result) -> str:
        if hasattr(result, "text"):
            return strip_asr_control_tokens(str(result.text))
        if isinstance(result, dict):
            return strip_asr_control_tokens(str(result.get("text") or ""))
        return strip_asr_control_tokens(str(result or ""))

    def _notify(self, progress: float, message: str) -> None:
        if self.progress_callback is None:
            return
        try:
            self.progress_callback(min(max(progress, 0.0), 1.0), message)
        except Exception:
            pass


class AssistedASRBackend(ASRBackend):
    """Run a primary backend and keep open-source candidate transcripts."""

    name = "assisted"

    def __init__(
        self,
        primary: ASRBackend,
        assistants: Iterable[ASRBackend],
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> None:
        self.primary = primary
        self.assistants = list(assistants)
        self.progress_callback = progress_callback
        self.alternatives: list[dict[str, object]] = []

    def set_progress_callback(
        self,
        callback: Callable[[float, str], None] | None,
    ) -> None:
        self.progress_callback = callback
        for backend in [self.primary, *self.assistants]:
            setter = getattr(backend, "set_progress_callback", None)
            if setter:
                setter(callback)

    def transcribe(self, audio_path: str | None = None) -> list[Segment]:
        self.alternatives = []
        primary_segments = self.primary.transcribe(audio_path)
        for assistant in self.assistants:
            backend_name = getattr(assistant, "name", assistant.__class__.__name__)
            try:
                self._notify(0.96, f"正在用开源候选模型复核：{backend_name}...")
                assistant_segments = assistant.transcribe(audio_path)
            except Exception as exc:
                self.alternatives.append(
                    {
                        "backend": backend_name,
                        "status": "failed",
                        "error": str(exc),
                    }
                )
                continue
            self.alternatives.append(
                {
                    "backend": backend_name,
                    "status": "ok",
                    "transcript": "".join(segment.display_text() for segment in assistant_segments),
                    "segments": [
                        {
                            "start": segment.start,
                            "end": segment.end,
                            "text": segment.display_text(),
                        }
                        for segment in assistant_segments
                    ],
                }
            )
        return primary_segments

    def _notify(self, progress: float, message: str) -> None:
        if self.progress_callback is None:
            return
        try:
            self.progress_callback(min(max(progress, 0.0), 1.0), message)
        except Exception:
            pass


class CollaborativeASRBackend(ASRBackend):
    """A multi-agent ASR coordinator with a strong primary and reviewers.

    The primary backend owns the first transcript. Reviewer backends produce
    candidate transcripts for later arbitration, repair, and active-learning
    logging. If the primary fails entirely, the first successful reviewer
    becomes the fallback transcript.
    """

    name = "dolphin_multiagent"

    def __init__(
        self,
        primary: ASRBackend,
        reviewers: Iterable[ASRBackend],
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> None:
        self.primary = primary
        self.reviewers = list(reviewers)
        self.progress_callback = progress_callback
        self.alternatives: list[dict[str, object]] = []
        self.agent_trace: list[dict[str, object]] = []

    def set_progress_callback(
        self,
        callback: Callable[[float, str], None] | None,
    ) -> None:
        self.progress_callback = callback
        for backend in [self.primary, *self.reviewers]:
            setter = getattr(backend, "set_progress_callback", None)
            if setter:
                setter(callback)

    def transcribe(self, audio_path: str | None = None) -> list[Segment]:
        self.alternatives = []
        self.agent_trace = []
        primary_name = getattr(self.primary, "name", self.primary.__class__.__name__)

        try:
            self._notify(0.05, "ASR 专家智能体正在识别：Dolphin 上海话模型...")
            primary_segments = self.primary.transcribe(audio_path)
            primary_text = "".join(segment.display_text() for segment in primary_segments)
            self.agent_trace.append(
                {
                    "agent": "Dolphin ASR 专家",
                    "role": "主识别",
                    "backend": primary_name,
                    "status": "ok",
                    "output_chars": len(primary_text),
                    "summary": "产出上海话主识别文本。",
                }
            )
        except Exception as exc:
            self.agent_trace.append(
                {
                    "agent": "Dolphin ASR 专家",
                    "role": "主识别",
                    "backend": primary_name,
                    "status": "failed",
                    "error": str(exc),
                    "summary": "主识别失败，交给复核智能体兜底。",
                }
            )
            fallback = self._run_reviewers(audio_path, allow_fallback=True)
            if fallback is not None:
                return fallback
            raise

        self._run_reviewers(audio_path, allow_fallback=False)
        return primary_segments

    def _run_reviewers(self, audio_path: str | None, allow_fallback: bool) -> list[Segment] | None:
        fallback_segments: list[Segment] | None = None
        for reviewer in self.reviewers:
            backend_name = getattr(reviewer, "name", reviewer.__class__.__name__)
            try:
                self._notify(0.88, f"复核智能体正在检查候选：{backend_name}...")
                reviewer_segments = reviewer.transcribe(audio_path)
            except Exception as exc:
                self.alternatives.append(
                    {
                        "backend": backend_name,
                        "status": "failed",
                        "error": str(exc),
                    }
                )
                self.agent_trace.append(
                    {
                        "agent": "吴语识别复核智能体",
                        "role": "候选复核/兜底",
                        "backend": backend_name,
                        "status": "failed",
                        "error": str(exc),
                        "summary": "候选识别失败，保留 Dolphin 主输出。",
                    }
                )
                continue

            transcript = "".join(segment.display_text() for segment in reviewer_segments)
            self.alternatives.append(
                {
                    "backend": backend_name,
                    "status": "ok",
                    "transcript": transcript,
                    "segments": [
                        {
                            "start": segment.start,
                            "end": segment.end,
                            "text": segment.display_text(),
                        }
                        for segment in reviewer_segments
                    ],
                }
            )
            self.agent_trace.append(
                {
                    "agent": "吴语识别复核智能体",
                    "role": "候选复核/兜底",
                    "backend": backend_name,
                    "status": "ok",
                    "output_chars": len(transcript),
                    "summary": "产出第二视角候选，用于仲裁和主动学习。",
                }
            )
            if allow_fallback and fallback_segments is None and transcript.strip():
                fallback_segments = reviewer_segments
        return fallback_segments

    def _notify(self, progress: float, message: str) -> None:
        if self.progress_callback is None:
            return
        try:
            self.progress_callback(min(max(progress, 0.0), 1.0), message)
        except Exception:
            pass


def make_backend(
    backend: str,
    model_name: str | None = None,
    local_files_only: bool = False,
    chunk_seconds: float = 15.0,
    vad_enabled: bool | None = None,
    max_speech_region_seconds: float | None = None,
    assist_backend: str | None = None,
    assist_model: str | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
) -> ASRBackend:
    if backend == "dolphin_multiagent":
        primary = _make_single_backend(
            "dolphin",
            model_name=model_name or "small.cn",
            local_files_only=local_files_only,
            chunk_seconds=chunk_seconds,
            vad_enabled=vad_enabled,
            max_speech_region_seconds=max_speech_region_seconds,
            progress_callback=progress_callback,
        )
        reviewer = _make_single_backend(
            "whisper_medium_wu",
            model_name=assist_model,
            local_files_only=local_files_only,
            chunk_seconds=chunk_seconds,
            vad_enabled=vad_enabled,
            max_speech_region_seconds=max_speech_region_seconds,
            progress_callback=progress_callback,
        )
        return CollaborativeASRBackend(primary, [reviewer], progress_callback=progress_callback)

    if backend == "hybrid":
        primary = _make_single_backend(
            "whisper",
            model_name=model_name,
            local_files_only=local_files_only,
            chunk_seconds=chunk_seconds,
            vad_enabled=vad_enabled,
            max_speech_region_seconds=max_speech_region_seconds,
            progress_callback=progress_callback,
        )
        assistant = _make_single_backend(
            "dolphin",
            model_name=assist_model or "small.cn",
            local_files_only=local_files_only,
            chunk_seconds=chunk_seconds,
            vad_enabled=vad_enabled,
            max_speech_region_seconds=max_speech_region_seconds,
            progress_callback=progress_callback,
        )
        return AssistedASRBackend(primary, [assistant], progress_callback=progress_callback)

    primary = _make_single_backend(
        backend,
        model_name=model_name,
        local_files_only=local_files_only,
        chunk_seconds=chunk_seconds,
        vad_enabled=vad_enabled,
        max_speech_region_seconds=max_speech_region_seconds,
        progress_callback=progress_callback,
    )
    if not assist_backend or assist_backend == "none" or backend == "mock":
        return primary
    assistant = _make_single_backend(
        assist_backend,
        model_name=assist_model,
        local_files_only=local_files_only,
        chunk_seconds=chunk_seconds,
        vad_enabled=vad_enabled,
        max_speech_region_seconds=max_speech_region_seconds,
        progress_callback=progress_callback,
    )
    return AssistedASRBackend(primary, [assistant], progress_callback=progress_callback)


def _make_single_backend(
    backend: str,
    model_name: str | None = None,
    local_files_only: bool = False,
    chunk_seconds: float = 15.0,
    vad_enabled: bool | None = None,
    max_speech_region_seconds: float | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
) -> ASRBackend:
    if backend == "mock":
        return MockASRBackend()
    if backend == "whisper":
        return WhisperTransformersBackend(
            model_name=model_name or "openai/whisper-small",
            local_files_only=local_files_only,
            chunk_seconds=chunk_seconds,
            vad_enabled=vad_enabled,
            max_speech_region_seconds=max_speech_region_seconds,
            progress_callback=progress_callback,
        )
    if backend == "funasr":
        return FunASRBackend(
            model_name=model_name or "iic/SenseVoiceSmall",
            local_files_only=local_files_only,
            max_speech_region_seconds=max_speech_region_seconds,
            progress_callback=progress_callback,
        )
    if backend == "dolphin":
        return DolphinBackend(
            model_name=model_name or "small.cn",
            local_files_only=local_files_only,
            progress_callback=progress_callback,
        )
    if backend == "whisper_medium_wu":
        return WhisperMediumWuBackend(
            model_name=model_name,
            local_files_only=local_files_only,
            progress_callback=progress_callback,
        )
    raise ValueError(f"Unknown backend: {backend}")


class normalized_temp_wav:
    def __init__(self, path: Path, target_sampling_rate: int) -> None:
        self.path = path
        self.target_sampling_rate = target_sampling_rate
        self.temp_path: Path | None = None

    def __enter__(self) -> Path:
        import soundfile as sf

        audio_input = load_audio_for_pipeline(self.path, self.target_sampling_rate)
        if isinstance(audio_input, str):
            return Path(audio_input)
        handle = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        handle.close()
        self.temp_path = Path(handle.name)
        sf.write(
            str(self.temp_path),
            audio_input["raw"],
            int(audio_input["sampling_rate"]),
            subtype="PCM_16",
        )
        return self.temp_path

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self.temp_path is not None:
            self.temp_path.unlink(missing_ok=True)


def load_audio_for_pipeline(path: Path, target_sampling_rate: int):
    """Load audio without requiring an ffmpeg executable.

    WAV/FLAC files are handled by soundfile. If the extension and actual
    container disagree, such as an M4A file renamed to .flac, fall back to the
    bundled imageio-ffmpeg binary and return a decoded float32 waveform.
    """

    try:
        import numpy as np
        import soundfile as sf
    except ImportError:
        return str(path)

    try:
        array, sampling_rate = sf.read(str(path), dtype="float32")
    except Exception as soundfile_error:
        try:
            return _load_audio_with_ffmpeg(path, target_sampling_rate)
        except Exception as ffmpeg_error:
            raise RuntimeError(
                f"无法读取音频文件：{path}。soundfile 报错：{soundfile_error}；"
                f"ffmpeg 兜底也失败：{ffmpeg_error}"
            ) from ffmpeg_error
    if getattr(array, "ndim", 1) > 1:
        array = array.mean(axis=1)
    if sampling_rate != target_sampling_rate:
        old_positions = np.linspace(0.0, 1.0, num=len(array), endpoint=False)
        new_length = max(1, round(len(array) * target_sampling_rate / sampling_rate))
        new_positions = np.linspace(0.0, 1.0, num=new_length, endpoint=False)
        array = np.interp(new_positions, old_positions, array).astype("float32")
        sampling_rate = target_sampling_rate
    return {"raw": array, "sampling_rate": sampling_rate}


def _load_audio_with_ffmpeg(path: Path, target_sampling_rate: int) -> dict[str, object]:
    import subprocess

    import imageio_ffmpeg
    import numpy as np

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-f",
        "f32le",
        "-ac",
        "1",
        "-ar",
        str(target_sampling_rate),
        "pipe:1",
    ]
    completed = subprocess.run(command, capture_output=True, check=False)
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(stderr or f"ffmpeg exited with code {completed.returncode}")
    audio = np.frombuffer(completed.stdout, dtype=np.float32)
    if audio.size == 0:
        raise RuntimeError("ffmpeg decoded an empty audio stream")
    return {"raw": audio, "sampling_rate": target_sampling_rate}
