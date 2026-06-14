from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a gender-aware, speaker-balanced ASR training manifest."
    )
    parser.add_argument("--base-train", default="data/splits/train_direct_mandarin.jsonl")
    parser.add_argument(
        "--video-train",
        default="outputs/video_pretraining/pretrain_manifests/video_direct_mandarin/train.jsonl",
    )
    parser.add_argument(
        "--dev-manifest",
        default="outputs/video_pretraining/pretrain_manifests/combined_direct_mandarin/dev.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/video_pretraining/pretrain_manifests/gender_speaker_balanced_direct_mandarin",
    )
    parser.add_argument(
        "--female-to-male-ratio",
        type=float,
        default=1.5,
        help="Target female:male row ratio for the public-data portion.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base_rows = read_jsonl(Path(args.base_train))
    video_rows = read_jsonl(Path(args.video_train))
    dev_rows = read_jsonl(Path(args.dev_manifest))
    rng = random.Random(args.seed)

    by_gender: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in base_rows:
        by_gender[str(row.get("gender") or "unknown").lower()].append(row)
    if not by_gender["male"] or not by_gender["female"]:
        raise ValueError("Both male and female rows are required for gender-aware balancing.")

    known_total = len(by_gender["male"]) + len(by_gender["female"])
    target_male = round(known_total / (1.0 + args.female_to_male_ratio))
    target_female = known_total - target_male
    balanced_base = sample_speaker_balanced(by_gender["male"], target_male, rng)
    balanced_base.extend(sample_speaker_balanced(by_gender["female"], target_female, rng))
    rng.shuffle(balanced_base)

    train_rows = balanced_base + video_rows
    rng.shuffle(train_rows)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "train.jsonl", train_rows)
    write_jsonl(output_dir / "dev.jsonl", dev_rows)

    summary = {
        "base_input_rows": len(base_rows),
        "video_rows": len(video_rows),
        "train_rows": len(train_rows),
        "dev_rows": len(dev_rows),
        "target_female_to_male_ratio": args.female_to_male_ratio,
        "before_gender": dict(Counter(str(row.get("gender") or "unknown") for row in base_rows)),
        "after_gender": dict(Counter(str(row.get("gender") or "unknown") for row in train_rows)),
        "before_speakers": speaker_counts(base_rows),
        "after_speakers": speaker_counts(balanced_base),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def sample_speaker_balanced(
    rows: list[dict[str, Any]], target_count: int, rng: random.Random
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("speaker_id") or "unknown")].append(row)
    for speaker_rows in grouped.values():
        rng.shuffle(speaker_rows)

    speakers = sorted(grouped)
    positions = {speaker: 0 for speaker in speakers}
    selected: list[dict[str, Any]] = []
    while len(selected) < target_count:
        cycle = speakers[:]
        rng.shuffle(cycle)
        for speaker in cycle:
            if len(selected) >= target_count:
                break
            speaker_rows = grouped[speaker]
            position = positions[speaker]
            selected.append(dict(speaker_rows[position % len(speaker_rows)]))
            positions[speaker] = position + 1
    return selected


def speaker_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get("speaker_id") or "unknown") for row in rows).items()))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
