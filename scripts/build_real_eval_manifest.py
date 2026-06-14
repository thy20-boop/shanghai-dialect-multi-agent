from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Build a real noisy eval manifest from online video test outputs.")
    parser.add_argument("--results", default=str(root / "outputs/online_video_tests/results.jsonl"))
    parser.add_argument("--output", default=str(root / "data/real_eval_manifest.jsonl"))
    parser.add_argument("--ocr-references", default=str(root / "outputs/hardsub_ocr/references.jsonl"))
    parser.add_argument("--include-ocr", action="store_true", help="Include OCR candidates when no manual reference exists.")
    parser.add_argument("--min-reference-chars", type=int, default=8)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> int:
    args = parse_args()
    rows = read_jsonl(Path(args.results))
    ocr_rows = read_jsonl(Path(args.ocr_references))
    ocr_references = {
        str(row["id"]): str(row.get("ocr_reference") or "").strip()
        for row in ocr_rows
        if row.get("id") and row.get("ocr_reference")
    }
    output_rows: list[dict[str, Any]] = []

    for row in rows:
        sample = row.get("sample", {})
        reference_source = "manual"
        reference = str(sample.get("reference") or "").strip()
        if not reference and args.include_ocr:
            reference = ocr_references.get(str(sample.get("id")), "")
            reference_source = "ocr"
        if len(reference) < args.min_reference_chars:
            continue
        product = row.get("product", {})
        transcript = product.get("draft_dialect_transcript") or product.get("dialect_transcript")
        mandarin = product.get("draft_mandarin") or product.get("mandarin")
        output_rows.append(
            {
                "id": sample.get("id"),
                "audio": row.get("clip_audio"),
                "reference": reference,
                "reference_source": reference_source,
                "transcript": transcript,
                "mandarin_translation": mandarin,
                "source_url": sample.get("url"),
                "title": sample.get("title"),
                "start": sample.get("start", 0),
                "duration": sample.get("duration", 0),
                "current_status": product.get("status"),
                "current_transcript": transcript,
                "current_mandarin": mandarin,
            }
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for row in output_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(output_rows)} referenced noisy samples to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
