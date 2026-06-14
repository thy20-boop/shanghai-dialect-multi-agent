from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Filter batch ASR predictions into high-confidence pseudo-label candidates."
    )
    parser.add_argument("--predictions", required=True, help="JSONL produced by ganagent.cli batch.")
    parser.add_argument("--output", required=True, help="Output pseudo-label manifest.")
    parser.add_argument("--max-suspicion-count", type=int, default=0)
    parser.add_argument("--min-dialect-score", type=float, default=0.25)
    parser.add_argument("--require-dialect", action="store_true")
    args = parser.parse_args()

    rows = read_jsonl(Path(args.predictions))
    kept: list[dict[str, Any]] = []
    for row in rows:
        if int(row.get("suspicion_count", 0)) > args.max_suspicion_count:
            continue
        if args.require_dialect and float(row.get("dialect_score", 0.0)) < args.min_dialect_score:
            continue
        audio = row.get("audio")
        transcript = row.get("transcript")
        if not audio or not transcript:
            continue
        kept.append(
            {
                "audio": audio,
                "text": transcript,
                "source": "pseudo_label",
                "dialect_score": row.get("dialect_score"),
                "repair_count": row.get("repair_count", 0),
            }
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output, kept)
    print(f"Kept {len(kept)} of {len(rows)} predictions -> {output}")
    return 0


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
