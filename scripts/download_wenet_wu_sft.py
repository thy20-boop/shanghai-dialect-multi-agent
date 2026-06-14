from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from huggingface_hub import snapshot_download


REPO_ID = "ASLP-lab/WenetSpeech-Wu-Speech-Generation"


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", required=True)
    args = parser.parse_args()

    runtime_root = Path(args.runtime_root).resolve()
    download_dir = runtime_root / "downloads" / "WenetSpeech-Wu-Speech-Generation"
    model_dir = runtime_root / "models" / "CosyVoice2-Wu-SFT-runtime"
    download_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=REPO_ID,
        local_dir=download_dir,
        allow_patterns=["CosyVoice2/*", "CosyVoice2-Wu-SFT/SFT.pt"],
        ignore_patterns=["CosyVoice2/llm.pt"],
    )

    base_dir = download_dir / "CosyVoice2"
    sft_weight = download_dir / "CosyVoice2-Wu-SFT" / "SFT.pt"
    if not (base_dir / "cosyvoice2.yaml").exists() or not sft_weight.exists():
        raise RuntimeError("WenetSpeech-Wu model download is incomplete.")

    for source in base_dir.rglob("*"):
        if source.is_file():
            link_or_copy(source, model_dir / source.relative_to(base_dir))
    link_or_copy(sft_weight, model_dir / "llm.pt")
    print(model_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
