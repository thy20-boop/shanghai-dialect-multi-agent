from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil


AUDIO_SUFFIXES = (".wav", ".flac", ".mp3", ".ogg", ".m4a")


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def find_recording(recording_dir: Path, item_id: str) -> Path | None:
    for suffix in AUDIO_SUFFIXES:
        candidate = recording_dir / f"{item_id}{suffix}"
        if candidate.exists():
            return candidate
    return None


def build_curriculum(evalset: Path, recording_dir: Path, output_dir: Path, speaker: str) -> dict:
    rows = read_jsonl(evalset)
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict] = []
    missing: list[str] = []
    for row in rows:
        item_id = row["id"]
        source = find_recording(recording_dir, item_id)
        if source is None:
            missing.append(item_id)
            continue
        target = audio_dir / f"{item_id}{source.suffix.lower()}"
        shutil.copy2(source, target)
        manifest_rows.append(
            {
                "id": item_id,
                "audio": str(target.resolve()),
                "text": row["wu_script"],
                "speaker_id": speaker,
                "target_terms": row.get("target_terms", []),
                "domain": row.get("domain", ""),
            }
        )

    manifest_path = output_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    gpt_list_path = output_dir / "gpt_sovits.list"
    with gpt_list_path.open("w", encoding="utf-8") as handle:
        for row in manifest_rows:
            handle.write(f"{row['audio']}|{row['speaker_id']}|zh|{row['text']}\n")

    summary = {
        "evalset": str(evalset),
        "recording_dir": str(recording_dir),
        "output_dir": str(output_dir),
        "item_count": len(manifest_rows),
        "missing_ids": missing,
        "manifest": str(manifest_path),
        "gpt_sovits_list": str(gpt_list_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a high-risk Wu TTS curriculum from teacher recordings.")
    parser.add_argument("--evalset", default="data/high_risk_wu_tts_eval.jsonl")
    parser.add_argument("--recording-dir", default="data/high_risk_wu_tts_recordings")
    parser.add_argument("--output-dir", default="outputs/wu_tts_curriculum")
    parser.add_argument("--speaker", default="wu_teacher")
    args = parser.parse_args()

    summary = build_curriculum(
        evalset=Path(args.evalset),
        recording_dir=Path(args.recording_dir),
        output_dir=Path(args.output_dir),
        speaker=args.speaker,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
