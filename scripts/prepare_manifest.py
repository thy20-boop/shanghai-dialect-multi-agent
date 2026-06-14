from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build ASR JSONL manifest from a TSV/CSV transcript file.")
    parser.add_argument("--transcripts", required=True, help="CSV/TSV/JSONL with audio and text columns.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--audio-column", default="audio")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--delimiter", default=None, help="Override delimiter, e.g. tab or comma.")
    args = parser.parse_args()

    input_path = Path(args.transcripts)
    rows = read_rows(input_path, args.delimiter)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            audio = row.get(args.audio_column)
            text = row.get(args.text_column)
            if not audio or not text:
                continue
            f.write(json.dumps({"audio": audio, "text": text}, ensure_ascii=False) + "\n")
            kept += 1
    print(f"Wrote {kept} examples to {output_path}")
    return 0


def read_rows(path: Path, delimiter: str | None) -> list[dict[str, str]]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    if delimiter == "tab":
        delimiter = "\t"
    elif delimiter == "comma":
        delimiter = ","
    elif delimiter is None:
        delimiter = "\t" if path.suffix.lower() == ".tsv" else ","

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        return list(reader)


if __name__ == "__main__":
    raise SystemExit(main())
