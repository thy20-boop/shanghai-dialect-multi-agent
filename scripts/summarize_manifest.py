from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize an ASR JSONL manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    rows = read_jsonl(Path(args.manifest))
    text_lengths = [len(str(row.get("text") or row.get("reference") or "")) for row in rows]
    missing_audio = sum(1 for row in rows if row.get("audio") and not Path(str(row["audio"])).exists())
    summary = {
        "manifest": args.manifest,
        "sample_count": len(rows),
        "missing_audio_count": missing_audio,
        "avg_text_length": round(sum(text_lengths) / len(text_lengths), 2) if text_lengths else 0.0,
        "min_text_length": min(text_lengths) if text_lengths else 0,
        "max_text_length": max(text_lengths) if text_lengths else 0,
        "first_examples": rows[:3],
    }

    text = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text)
    return 0


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
