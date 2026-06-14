from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any


TITLE_RE = re.compile(r"^\s*[0-9一二三四五六七八九十]+[\s、.．-]*[\u4e00-\u9fff]{1,8}\s*$")
CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Clean hard-subtitle cues, split by source video, and build ASR pretraining manifests."
    )
    parser.add_argument("--cues", default="outputs/video_pretraining/hardsub/cues.jsonl")
    parser.add_argument("--output-dir", default="outputs/video_pretraining/pretrain_manifests")
    parser.add_argument("--base-wu-train", default="data/splits/train.jsonl")
    parser.add_argument("--base-wu-dev", default="data/splits/dev.jsonl")
    parser.add_argument("--base-mandarin-train", default="data/splits/train_direct_mandarin.jsonl")
    parser.add_argument("--base-mandarin-dev", default="data/splits/dev_direct_mandarin.jsonl")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-chinese", type=int, default=4)
    parser.add_argument("--min-duration", type=float, default=0.8)
    parser.add_argument("--max-duration", type=float, default=15.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_cues = read_jsonl(Path(args.cues))
    kept_cues, rejected = clean_cues(
        raw_cues,
        min_chinese=args.min_chinese,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
    )

    split_source_ids = split_sources(kept_cues, args.train_ratio, args.dev_ratio, args.seed)
    cue_splits = {
        split: [cue for cue in kept_cues if cue["source_id"] in source_ids]
        for split, source_ids in split_source_ids.items()
    }

    write_jsonl(output_dir / "filtered_cues.jsonl", kept_cues)
    write_jsonl(output_dir / "rejected_cues.jsonl", rejected)

    video_wu_dir = output_dir / "video_wu"
    video_mandarin_dir = output_dir / "video_direct_mandarin"
    combined_wu_dir = output_dir / "combined_wu"
    combined_mandarin_dir = output_dir / "combined_direct_mandarin"
    for path in [video_wu_dir, video_mandarin_dir, combined_wu_dir, combined_mandarin_dir]:
        path.mkdir(parents=True, exist_ok=True)

    video_wu = write_video_manifests(video_wu_dir, cue_splits, text_key="wu_text", target_type="wu_transcript")
    video_mandarin = write_video_manifests(
        video_mandarin_dir,
        cue_splits,
        text_key="mandarin",
        target_type="direct_mandarin",
    )

    combined_wu = combine_base_and_video(
        combined_wu_dir,
        Path(args.base_wu_train),
        Path(args.base_wu_dev),
        video_wu,
    )
    combined_mandarin = combine_base_and_video(
        combined_mandarin_dir,
        Path(args.base_mandarin_train),
        Path(args.base_mandarin_dev),
        video_mandarin,
    )

    summary = {
        "input_cues": len(raw_cues),
        "kept_cues": len(kept_cues),
        "rejected_cues": len(rejected),
        "rejected_reasons": dict(Counter(item["reject_reason"] for item in rejected)),
        "source_splits": {split: sorted(ids) for split, ids in split_source_ids.items()},
        "video_wu": video_wu["counts"],
        "video_direct_mandarin": video_mandarin["counts"],
        "combined_wu": combined_wu,
        "combined_direct_mandarin": combined_mandarin,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def clean_cues(
    cues: list[dict[str, Any]],
    *,
    min_chinese: int,
    min_duration: float,
    max_duration: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen: set[tuple[str, float, float, str, str]] = set()

    for cue in cues:
        wu_text = normalize_text(str(cue.get("wu_text") or ""))
        mandarin = normalize_text(str(cue.get("mandarin") or ""))
        start = float(cue.get("start") or 0.0)
        end = float(cue.get("end") or 0.0)
        duration = max(0.0, end - start)

        reason = reject_reason(wu_text, mandarin, duration, min_chinese, min_duration, max_duration)
        key = (str(cue.get("source_id") or ""), start, end, wu_text, mandarin)
        if key in seen:
            reason = reason or "duplicate"
        if reason:
            rejected.append({**cue, "wu_text": wu_text, "mandarin": mandarin, "reject_reason": reason})
            continue

        seen.add(key)
        kept.append(
            {
                **cue,
                "wu_text": wu_text,
                "mandarin": mandarin,
                "duration": round(duration, 3),
            }
        )
    return kept, rejected


def reject_reason(
    wu_text: str,
    mandarin: str,
    duration: float,
    min_chinese: int,
    min_duration: float,
    max_duration: float,
) -> str | None:
    if duration < min_duration:
        return "too_short_audio"
    if duration > max_duration:
        return "too_long_audio"
    if chinese_count(wu_text) < min_chinese or chinese_count(mandarin) < min_chinese:
        return "too_little_chinese"
    if TITLE_RE.match(wu_text) and TITLE_RE.match(mandarin):
        return "title_like"
    if "上海话情景对话" in wu_text or "上海话情景对话" in mandarin:
        return "series_title"
    return None


def split_sources(
    cues: list[dict[str, Any]],
    train_ratio: float,
    dev_ratio: float,
    seed: int,
) -> dict[str, set[str]]:
    source_ids = sorted({str(cue["source_id"]) for cue in cues})
    random.Random(seed).shuffle(source_ids)
    train_end = int(len(source_ids) * train_ratio)
    dev_end = train_end + max(1, int(len(source_ids) * dev_ratio))
    return {
        "train": set(source_ids[:train_end]),
        "dev": set(source_ids[train_end:dev_end]),
        "test": set(source_ids[dev_end:]),
    }


def write_video_manifests(
    output_dir: Path,
    cue_splits: dict[str, list[dict[str, Any]]],
    *,
    text_key: str,
    target_type: str,
) -> dict[str, Any]:
    paths: dict[str, Path] = {}
    counts: dict[str, int] = {}
    for split, cues in cue_splits.items():
        rows = [
            {
                "audio": cue["audio"],
                "text": cue[text_key],
                "source_id": cue["source_id"],
                "source_video": cue["source_video"],
                "start": cue["start"],
                "end": cue["end"],
                "duration": cue["duration"],
                "target_type": target_type,
                "source_dataset": "local_hardsub_video",
            }
            for cue in cues
            if cue.get(text_key)
        ]
        path = output_dir / f"{split}.jsonl"
        write_jsonl(path, rows)
        paths[split] = path
        counts[split] = len(rows)
    return {"paths": paths, "counts": counts}


def combine_base_and_video(
    output_dir: Path,
    base_train: Path,
    base_dev: Path,
    video_manifest: dict[str, Any],
) -> dict[str, int]:
    base_train_rows = read_jsonl(base_train)
    base_dev_rows = read_jsonl(base_dev)
    video_train_rows = read_jsonl(video_manifest["paths"]["train"])
    video_dev_rows = read_jsonl(video_manifest["paths"]["dev"])
    video_test_rows = read_jsonl(video_manifest["paths"]["test"])

    splits = {
        "train": base_train_rows + video_train_rows,
        "dev": base_dev_rows + video_dev_rows,
        "test": video_test_rows,
    }
    counts: dict[str, int] = {}
    for split, rows in splits.items():
        path = output_dir / f"{split}.jsonl"
        write_jsonl(path, rows)
        counts[split] = len(rows)
    return counts


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", text.strip())


def chinese_count(text: str) -> int:
    return len(CHINESE_RE.findall(text))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
