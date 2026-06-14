from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ganagent.models import Segment
from ganagent.repair import RepairEngine, count_repair_actions


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert Shanghainese ASR manifests into direct audio-to-Mandarin training manifests."
    )
    parser.add_argument("--train-input", default="data/splits/train.jsonl")
    parser.add_argument("--dev-input", default="data/splits/dev.jsonl")
    parser.add_argument("--train-output", default="data/splits/train_direct_mandarin.jsonl")
    parser.add_argument("--dev-output", default="data/splits/dev_direct_mandarin.jsonl")
    parser.add_argument("--glossary", default="data/examples/shanghainese_glossary.json")
    args = parser.parse_args()

    engine = RepairEngine.from_file(args.glossary)
    train_summary = convert_manifest(Path(args.train_input), Path(args.train_output), engine)
    dev_summary = convert_manifest(Path(args.dev_input), Path(args.dev_output), engine)
    summary = {"train": train_summary, "dev": dev_summary}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def convert_manifest(input_path: Path, output_path: Path, engine: RepairEngine) -> dict[str, Any]:
    rows = []
    repair_counts = []
    changed = 0
    for row in read_jsonl(input_path):
        original = str(row.get("text") or "").strip()
        target, repairs = normalize_target(original, engine)
        if target != original:
            changed += 1
        repair_counts.append(count_repair_actions(repairs))
        new_row = dict(row)
        new_row["text"] = target
        new_row["source_text"] = original
        new_row["target_type"] = "direct_mandarin_pseudo"
        new_row["pseudo_repairs"] = repairs
        rows.append(new_row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    return {
        "input": str(input_path),
        "output": str(output_path),
        "rows": len(rows),
        "changed_rows": changed,
        "avg_repair_actions": round(sum(repair_counts) / max(len(repair_counts), 1), 4),
    }


def normalize_target(text: str, engine: RepairEngine) -> tuple[str, list[dict[str, Any]]]:
    segment = Segment(start=0.0, end=0.0, text=text)
    repairs = engine.repair_segments([segment])
    repaired = segment.display_text()
    mandarin, translation_repairs = engine.translate_to_mandarin_with_replacements(repaired)
    mandarin = cleanup_mandarin(mandarin)
    return mandarin, repairs + translation_repairs


def cleanup_mandarin(text: str) -> str:
    replacements = {
        "几个月月": "几个月",
        "找一找一找": "找一找",
        "什么什么": "什么",
        "在在": "在",
        "你好吗？吗": "你好吗？",
    }
    cleaned = text
    for wrong, right in replacements.items():
        cleaned = cleaned.replace(wrong, right)
    return cleaned.strip()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
