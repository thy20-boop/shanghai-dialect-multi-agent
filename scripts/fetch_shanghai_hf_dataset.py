from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


DEFAULT_DATASET = "TingChen-ppmc/Shanghai_Dialect_Conversational_Speech_Corpus"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch the public Shanghai dialect HF dataset into an ASR JSONL manifest."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", required=True, help="Output JSONL manifest path.")
    parser.add_argument("--audio-dir", required=True, help="Directory to write WAV files.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    try:
        from datasets import Audio, DownloadConfig, load_dataset
    except ImportError as exc:
        raise RuntimeError("Install dependencies first: python -m pip install datasets") from exc

    download_config = DownloadConfig(local_files_only=args.local_files_only)
    dataset = load_dataset(
        args.dataset,
        split=args.split,
        download_config=download_config,
    )
    if "audio" in dataset.features:
        dataset = dataset.cast_column("audio", Audio(decode=False))

    audio_dir = Path(args.audio_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with output.open("w", encoding="utf-8") as f:
        for index, item in enumerate(dataset):
            if args.max_samples is not None and written >= args.max_samples:
                break
            text = extract_text(item)
            audio = item.get("audio")
            if not text or not audio:
                continue

            wav_name = f"shanghai_{index:06d}.wav"
            wav_path = audio_dir / wav_name
            write_audio_file(audio, wav_path)

            record = {
                "audio": str(wav_path.as_posix()),
                "text": text,
                "speaker_id": item.get("speaker_id"),
                "gender": item.get("gender"),
                "source_dataset": args.dataset,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} examples to {output}")
    print(f"Audio directory: {audio_dir}")
    return 0


def extract_text(item: dict[str, Any]) -> str | None:
    for key in ("transcription", "text", "sentence", "transcript"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def write_audio_file(audio: dict[str, Any], output_path: Path) -> None:
    source_path = audio.get("path")
    if source_path and Path(source_path).exists():
        shutil.copyfile(source_path, output_path)
        return

    raw_bytes = audio.get("bytes")
    if raw_bytes:
        output_path.write_bytes(raw_bytes)
        return

    if "array" in audio and "sampling_rate" in audio:
        try:
            import soundfile as sf
        except ImportError as exc:
            raise RuntimeError("Install soundfile to export decoded audio arrays.") from exc
        sf.write(output_path, audio["array"], int(audio["sampling_rate"]))
        return

    raise ValueError("Audio item has no path, bytes, or decoded array.")


if __name__ == "__main__":
    raise SystemExit(main())
