from __future__ import annotations

import io
import os
import random
import sys
import threading
import wave
from pathlib import Path

import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel


def default_runtime_root() -> Path:
    configured = os.environ.get("SHANGHAI_WU_RUNTIME")
    if configured:
        return Path(configured)
    legacy = Path(r"D:\wswu_runtime")
    if legacy.exists():
        return legacy
    local_app_data = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    return local_app_data / "ShanghaiDialectAgent" / "wswu_runtime"


RUNTIME_ROOT = default_runtime_root()
os.environ.setdefault("MODELSCOPE_CACHE", str(RUNTIME_ROOT / "modelscope_cache"))

COSYVOICE_DIR = Path(os.environ.get("WSWU_COSYVOICE_DIR", str(RUNTIME_ROOT / "CosyVoice")))
MODEL_DIR = Path(
    os.environ.get(
        "WSWU_MODEL_DIR",
        str(RUNTIME_ROOT / "models" / "CosyVoice2-Wu-SFT-runtime"),
    )
)
PROMPT_AUDIO = Path(os.environ.get("WSWU_PROMPT_AUDIO", str(RUNTIME_ROOT / "prompt_shanghai.wav")))
DEFAULT_PROMPT_TEXT = "最少辰光阿拉是做撒呃喃，有钞票就是到银行里保本保息。"
EXPERT_MODE = os.environ.get("WSWU_EXPERT_MODE", "zero_shot")
USE_FP16 = os.environ.get("WSWU_FP16", "0") == "1"
DEFAULT_INSTRUCTION = os.environ.get(
    "WSWU_INSTRUCTION",
    "这是一位上海人，用自然、清楚、平稳的上海话说",
)

sys.path.insert(0, str(COSYVOICE_DIR))
sys.path.insert(0, str(COSYVOICE_DIR / "third_party" / "Matcha-TTS"))

from cosyvoice.cli.cosyvoice import CosyVoice2  # noqa: E402


class SynthesisRequest(BaseModel):
    text: str
    prompt_text: str = DEFAULT_PROMPT_TEXT
    prompt_audio: str | None = None
    speed: float = 1.0
    instruction: str = DEFAULT_INSTRUCTION
    use_text_frontend: bool = True
    seed: int = 1986


app = FastAPI(title="WenetSpeech-Wu CosyVoice2 Expert")
_model: CosyVoice2 | None = None
_model_lock = threading.Lock()
_inference_lock = threading.Lock()


def get_model() -> CosyVoice2:
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                if not MODEL_DIR.exists():
                    raise RuntimeError(f"Wu model directory not found: {MODEL_DIR}")
                _model = CosyVoice2(
                    str(MODEL_DIR),
                    load_jit=False,
                    load_trt=False,
                    load_vllm=False,
                    fp16=torch.cuda.is_available() and USE_FP16,
                )
    return _model


def wav_bytes(audio: torch.Tensor, sample_rate: int) -> bytes:
    samples = audio.squeeze().clamp(-1, 1)
    pcm = (samples * 32767).to(torch.int16).cpu().numpy().tobytes()
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)
    return buffer.getvalue()


@app.get("/health")
def health() -> dict:
    frontend = getattr(getattr(_model, "frontend", None), "text_frontend", None)
    return {
        "status": "ok",
        "model": str(MODEL_DIR),
        "model_loaded": _model is not None,
        "cuda": torch.cuda.is_available(),
        "expert_mode": EXPERT_MODE,
        "fp16": USE_FP16,
        "text_frontend": frontend,
    }


@app.post("/tts")
def synthesize(request: SynthesisRequest) -> Response:
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text must not be empty")
    prompt_audio = Path(request.prompt_audio) if request.prompt_audio else PROMPT_AUDIO
    if not prompt_audio.exists():
        raise HTTPException(status_code=400, detail=f"prompt audio not found: {prompt_audio}")
    try:
        model = get_model()
        if request.use_text_frontend:
            text = model.frontend.text_normalize(
                text,
                split=False,
                text_frontend=True,
            )
        if not text.startswith("<|wuyu|>"):
            text = "<|wuyu|>" + text
        with _inference_lock:
            random.seed(request.seed)
            np.random.seed(request.seed % (2**32 - 1))
            torch.manual_seed(request.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(request.seed)
            if EXPERT_MODE == "instruct_prosody":
                chunks = list(
                    model.inference_instruct2(
                        text,
                        request.instruction.strip() or DEFAULT_INSTRUCTION,
                        str(prompt_audio),
                        stream=False,
                        speed=request.speed,
                        text_frontend=False,
                    )
                )
            else:
                chunks = list(
                    model.inference_zero_shot(
                        text,
                        request.prompt_text.strip() or DEFAULT_PROMPT_TEXT,
                        str(prompt_audio),
                        stream=False,
                        speed=request.speed,
                        text_frontend=False,
                    )
                )
        if not chunks:
            raise RuntimeError("CosyVoice2-Wu returned no audio")
        audio = torch.cat([chunk["tts_speech"].cpu() for chunk in chunks], dim=1)
        return Response(wav_bytes(audio, model.sample_rate), media_type="audio/wav")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
