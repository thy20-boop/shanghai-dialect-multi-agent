from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np


@dataclass(frozen=True)
class RecordingConfig:
    sample_rate: int = 16000
    channels: int = 1
    device: int | str | None = None
    frame_ms: int = 30
    max_seconds: float = 18.0
    min_speech_seconds: float = 0.45
    start_threshold: float = 0.012
    stop_threshold: float = 0.007
    silence_seconds: float = 0.85
    preroll_seconds: float = 0.35


@dataclass(frozen=True)
class RecordingResult:
    path: Path
    duration_seconds: float
    speech_started: bool
    peak_rms: float


def record_utterance(
    output_path: str | Path,
    *,
    config: RecordingConfig | None = None,
) -> RecordingResult:
    """Record one spoken turn from the microphone using a lightweight RMS VAD."""

    try:
        import sounddevice as sd
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError(
            "实时对话需要麦克风依赖：python -m pip install sounddevice soundfile numpy"
        ) from exc

    cfg = config or RecordingConfig()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame_size = max(1, int(cfg.sample_rate * cfg.frame_ms / 1000))
    max_frames = max(1, int(cfg.max_seconds * cfg.sample_rate / frame_size))
    silence_limit = max(1, int(cfg.silence_seconds * cfg.sample_rate / frame_size))
    preroll_limit = max(1, int(cfg.preroll_seconds * cfg.sample_rate / frame_size))
    min_speech_frames = max(1, int(cfg.min_speech_seconds * cfg.sample_rate / frame_size))

    preroll: deque[np.ndarray] = deque(maxlen=preroll_limit)
    recorded: list[np.ndarray] = []
    speech_started = False
    speech_frames = 0
    silent_frames = 0
    peak_rms = 0.0
    started_at = time.monotonic()

    with sd.InputStream(
        samplerate=cfg.sample_rate,
        channels=cfg.channels,
        dtype="float32",
        blocksize=frame_size,
        device=cfg.device,
    ) as stream:
        for _ in range(max_frames):
            frame, _ = stream.read(frame_size)
            mono = np.asarray(frame, dtype=np.float32)
            if mono.ndim > 1:
                mono = mono.mean(axis=1)
            rms = float(np.sqrt(np.mean(np.square(mono))) if mono.size else 0.0)
            peak_rms = max(peak_rms, rms)

            if not speech_started:
                preroll.append(mono.copy())
                if rms >= cfg.start_threshold:
                    speech_started = True
                    recorded.extend(preroll)
                    preroll.clear()
                continue

            recorded.append(mono.copy())
            speech_frames += 1
            if rms < cfg.stop_threshold:
                silent_frames += 1
            else:
                silent_frames = 0
            if speech_frames >= min_speech_frames and silent_frames >= silence_limit:
                break

    if recorded:
        audio = np.concatenate(recorded)
    else:
        audio = np.zeros(frame_size, dtype=np.float32)
    sf.write(str(output), audio, cfg.sample_rate)
    duration = len(audio) / float(cfg.sample_rate)
    if not speech_started:
        duration = time.monotonic() - started_at
    return RecordingResult(
        path=output,
        duration_seconds=round(duration, 3),
        speech_started=speech_started,
        peak_rms=round(peak_rms, 5),
    )


def list_input_devices() -> list[dict]:
    """Return available microphone-like input devices."""

    try:
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError(
            "列出麦克风需要依赖：python -m pip install sounddevice"
        ) from exc
    devices = sd.query_devices()
    rows: list[dict] = []
    for index, device in enumerate(devices):
        if int(device.get("max_input_channels", 0)) <= 0:
            continue
        rows.append(
            {
                "index": index,
                "name": str(device.get("name", "")),
                "hostapi": int(device.get("hostapi", -1)),
                "max_input_channels": int(device.get("max_input_channels", 0)),
                "default_samplerate": float(device.get("default_samplerate", 0.0)),
            }
        )
    return rows


def calibrate_noise_floor(
    *,
    seconds: float = 1.2,
    sample_rate: int = 16000,
    device: int | str | None = None,
) -> dict[str, float]:
    """Measure ambient RMS and suggest conservative VAD thresholds."""

    try:
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError(
            "麦克风校准需要依赖：python -m pip install sounddevice"
        ) from exc
    frames = max(1, int(seconds * sample_rate))
    audio = sd.rec(frames, samplerate=sample_rate, channels=1, dtype="float32", device=device)
    sd.wait()
    values = np.asarray(audio, dtype=np.float32).reshape(-1)
    rms = float(np.sqrt(np.mean(np.square(values))) if values.size else 0.0)
    p95 = float(np.percentile(np.abs(values), 95)) if values.size else 0.0
    start = max(0.010, rms * 3.2, p95 * 1.8)
    stop = max(0.006, rms * 2.0, p95 * 1.1)
    return {
        "noise_rms": round(rms, 6),
        "noise_p95": round(p95, 6),
        "start_threshold": round(start, 6),
        "stop_threshold": round(min(start * 0.85, stop), 6),
    }


def play_audio(path: str | Path) -> bool:
    """Play WAV/MP3 locally. Returns False if only an external player was opened."""

    audio_path = Path(path)
    if not audio_path.exists():
        raise FileNotFoundError(audio_path)
    try:
        import sounddevice as sd
        import soundfile as sf
    except ImportError:
        _open_with_system_player(audio_path)
        return False

    playable_path = audio_path
    cleanup_path: Path | None = None
    if audio_path.suffix.lower() != ".wav":
        cleanup_path = audio_path.with_suffix(".playback.wav")
        if not _convert_to_wav(audio_path, cleanup_path):
            return False
        playable_path = cleanup_path
    data, sample_rate = sf.read(str(playable_path), dtype="float32")
    sd.play(data, sample_rate)
    sd.wait()
    if cleanup_path and cleanup_path.exists():
        cleanup_path.unlink(missing_ok=True)
    return True


def _convert_to_wav(input_path: Path, output_path: Path) -> bool:
    import subprocess

    try:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        _open_with_system_player(input_path)
        return False
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(input_path),
            "-ac",
            "1",
            "-ar",
            "16000",
            str(output_path),
        ],
        check=True,
    )
    return True


def _open_with_system_player(path: Path) -> None:
    import os
    import sys
    import subprocess

    if sys.platform.startswith("win"):
        os.startfile(str(path))  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])
