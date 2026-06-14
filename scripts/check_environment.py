from __future__ import annotations

import importlib.util
import json
import platform
import sys


PACKAGES = [
    "datasets",
    "soundfile",
    "transformers",
    "torch",
    "accelerate",
    "peft",
    "imageio_ffmpeg",
    "streamlit",
    "pytest",
]


def main() -> int:
    status = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {name: bool(importlib.util.find_spec(name)) for name in PACKAGES},
    }
    print(json.dumps(status, ensure_ascii=False, indent=2))
    missing_core = [name for name in ("datasets", "soundfile") if not status["packages"][name]]
    if missing_core:
        print()
        print("Data download dependencies missing:")
        print("python -m pip install datasets soundfile")
    missing_asr = [name for name in ("transformers", "torch", "imageio_ffmpeg") if not status["packages"][name]]
    if missing_asr:
        print()
        print("Whisper/ASR dependencies missing:")
        print("python -m pip install -e .[asr]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
