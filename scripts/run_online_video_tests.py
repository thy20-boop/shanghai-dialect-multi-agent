from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ganagent.agent import ShanghaiDialectAgent
from ganagent.asr_backends import make_backend
from ganagent.evaluation import cer
from ganagent.product import build_translation_product
from ganagent.repair import RepairEngine


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Download short online Shanghai dialect clips and run the agent.")
    parser.add_argument("--manifest", default=str(root / "data/examples/online_video_samples.jsonl"))
    parser.add_argument("--output-dir", default=str(root / "outputs/online_video_tests"))
    parser.add_argument("--model", default="TingChen-ppmc/whisper-small-Shanghai")
    parser.add_argument("--glossary", default=str(root / "data/examples/shanghainese_glossary.json"))
    parser.add_argument("--ocr-references", default=str(root / "outputs/hardsub_ocr/references.jsonl"))
    parser.add_argument("--chunk-seconds", type=float, default=15.0)
    parser.add_argument("--online-model", action="store_true", help="Allow downloading model files.")
    parser.add_argument("--redownload", action="store_true", help="Download audio even if a raw file already exists.")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_ocr_reference_map(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    rows = read_jsonl(path)
    return {
        str(row["id"]): str(row.get("ocr_reference") or "").strip()
        for row in rows
        if row.get("id") and row.get("ocr_reference")
    }


def run_command(command: list[str], cwd: Path) -> None:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed ({completed.returncode}): {' '.join(command)}\n{completed.stdout}")


def find_ffmpeg() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def find_raw_audio(raw_dir: Path, sample_id: str) -> Path | None:
    candidates = [
        path
        for path in sorted(raw_dir.glob(f"{sample_id}.*"))
        if path.suffix not in {".part", ".ytdl"} and path.is_file()
    ]
    return candidates[0] if candidates else None


def download_audio(root: Path, raw_dir: Path, sample: dict[str, Any], redownload: bool) -> Path:
    sample_id = str(sample["id"])
    existing = find_raw_audio(raw_dir, sample_id)
    if existing and not redownload:
        return existing

    output_template = str(raw_dir / f"{sample_id}.%(ext)s")
    command = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--force-ipv4",
        "-f",
        "ba",
        "-o",
        output_template,
        str(sample["url"]),
    ]
    run_command(command, root)
    raw = find_raw_audio(raw_dir, sample_id)
    if raw is None:
        raise FileNotFoundError(f"yt-dlp finished but no raw audio was found for {sample_id}")
    return raw


def clip_name(sample: dict[str, Any]) -> str:
    start = int(float(sample.get("start", 0)))
    end = start + int(float(sample.get("duration", 20)))
    return f"{sample['id']}_{start:04d}_{end:04d}.wav"


def make_clip(root: Path, ffmpeg: str, output_dir: Path, raw_audio: Path, sample: dict[str, Any]) -> Path:
    wav_path = output_dir / clip_name(sample)
    command = [
        ffmpeg,
        "-y",
        "-ss",
        str(sample.get("start", 0)),
        "-t",
        str(sample.get("duration", 20)),
        "-i",
        str(raw_audio),
        "-ac",
        "1",
        "-ar",
        "16000",
        str(wav_path),
    ]
    run_command(command, root)
    return wav_path


def suspicion_dicts(result: Any) -> list[dict[str, Any]]:
    return [item.__dict__ for item in result.suspicions]


def render_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Online Video Test Results",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "| Sample | Clip | Status | Repairs | Suspicions | Mandarin output |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for row in rows:
        sample = row["sample"]
        product = row["product"]
        title = str(sample.get("title") or sample["id"]).replace("|", " ")
        url = sample.get("url", "")
        clip = f"{sample.get('start', 0)}s+{sample.get('duration', 20)}s"
        mandarin = str(product.get("mandarin", "")).replace("\n", " ").replace("|", " ")
        draft = str(product.get("draft_mandarin") or "").replace("\n", " ").replace("|", " ")
        scoring_text = str(
            product.get("draft_dialect_transcript")
            or product.get("draft_mandarin")
            or product.get("dialect_transcript")
            or product.get("mandarin")
            or ""
        )
        if len(mandarin) > 90:
            mandarin = mandarin[:87] + "..."
        if draft:
            draft_preview = draft[:87] + "..." if len(draft) > 90 else draft
            mandarin = f"{mandarin}<br><sub>草稿: {draft_preview}</sub>"
        reference_source = "参考字幕"
        reference = str(sample.get("reference") or "").replace("\n", " ").replace("|", " ")
        if not reference and sample.get("ocr_reference"):
            reference_source = "OCR候选"
            reference = str(sample.get("ocr_reference") or "").replace("\n", " ").replace("|", " ")
        if reference:
            score = cer(reference, scoring_text)
            if score > 0.6:
                product = {**product, "status_label": f"{product.get('status_label')} / 参考不合格"}
            mandarin = f"{mandarin}<br><sub>{reference_source} CER={score:.2f}: {reference[:70]}</sub>"
        lines.append(
            f"| [{title}]({url}) | {clip} | {product.get('status_label')} | "
            f"{product.get('repair_count')} | {product.get('suspicion_count')} | {mandarin} |"
        )
    lines.append("")
    lines.append("Note: raw video/audio is used only for local testing; the generated WAV clips are not redistributed.")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = Path(args.manifest)
    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    samples = read_jsonl(manifest)
    ocr_references = load_ocr_reference_map(Path(args.ocr_references))
    for sample in samples:
        if not sample.get("reference") and sample.get("id") in ocr_references:
            sample["ocr_reference"] = ocr_references[str(sample["id"])]
    ffmpeg = find_ffmpeg()

    prepared: list[tuple[dict[str, Any], Path, Path]] = []
    for sample in samples:
        raw_audio = download_audio(root, raw_dir, sample, redownload=args.redownload)
        wav_path = make_clip(root, ffmpeg, output_dir, raw_audio, sample)
        prepared.append((sample, raw_audio, wav_path))

    repair_engine = RepairEngine.from_file(args.glossary)
    backend = make_backend(
        "whisper",
        model_name=args.model,
        local_files_only=not args.online_model,
        chunk_seconds=args.chunk_seconds,
    )
    agent = ShanghaiDialectAgent(asr_backend=backend, repair_engine=repair_engine)

    rows: list[dict[str, Any]] = []
    for sample, raw_audio, wav_path in prepared:
        result = agent.run(audio_path=str(wav_path))
        product = build_translation_product(result)
        rows.append(
            {
                "sample": sample,
                "raw_audio": str(raw_audio),
                "clip_audio": str(wav_path),
                "product": product.as_dict(),
                "repairs": result.repairs,
                "suspicions": suspicion_dicts(result),
            }
        )
        print(f"{sample['id']}: {product.status_label} -> {product.mandarin}")

    write_jsonl(output_dir / "results.jsonl", rows)
    render_summary(output_dir / "summary.md", rows)
    print(f"Wrote {len(rows)} rows to {output_dir / 'results.jsonl'}")
    print(f"Wrote summary to {output_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
