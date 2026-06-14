from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description="Split an ASR JSONL manifest into train/dev/test files.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-count", type=int, default=None)
    parser.add_argument("--dev-count", type=int, default=None)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = read_jsonl(Path(args.manifest))
    random.Random(args.seed).shuffle(rows)

    if args.train_count is not None or args.dev_count is not None:
        if args.train_count is None or args.dev_count is None:
            raise ValueError("--train-count and --dev-count must be provided together.")
        if args.train_count + args.dev_count > len(rows):
            raise ValueError(
                f"Requested {args.train_count + args.dev_count} examples from a manifest with {len(rows)} rows."
            )
        train_end = args.train_count
        dev_end = train_end + args.dev_count
    else:
        train_end = int(len(rows) * args.train_ratio)
        dev_end = train_end + int(len(rows) * args.dev_ratio)

    splits = {
        "train": rows[:train_end],
        "dev": rows[train_end:dev_end],
    }
    if rows[dev_end:]:
        splits["test"] = rows[dev_end:]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, split_rows in splits.items():
        path = output_dir / f"{name}.jsonl"
        write_jsonl(path, split_rows)
        print(f"{name}: {len(split_rows)} -> {path}")
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
