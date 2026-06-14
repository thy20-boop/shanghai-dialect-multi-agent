from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


CREDIT_PATTERNS = ("@", "翻：", "翻:", "后制", "後製", "剪辑", "字幕")
SPEAKER_RE = re.compile(r"^[\u4e00-\u9fa5A-Za-z]{1,4}[：:]\s*")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Extract hard-subtitle reference candidates from online videos.")
    parser.add_argument("--manifest", default=str(root / "data/examples/online_video_samples.jsonl"))
    parser.add_argument("--output", default=str(root / "outputs/hardsub_ocr/references.jsonl"))
    parser.add_argument("--video-dir", default=str(root / "outputs/online_video_tests/video"))
    parser.add_argument("--frame-dir", default=str(root / "outputs/hardsub_ocr/frames"))
    parser.add_argument("--frame-step", type=float, default=5.0)
    parser.add_argument("--redownload", action="store_true")
    parser.add_argument("--min-confidence", type=float, default=0.70)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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


def video_path_for(video_dir: Path, sample_id: str) -> Path | None:
    for suffix in (".mp4", ".mkv", ".webm"):
        path = video_dir / f"{sample_id}{suffix}"
        if path.exists():
            return path
    matches = sorted(video_dir.glob(f"{sample_id}.*"))
    return matches[0] if matches else None


def download_video(root: Path, video_dir: Path, sample: dict[str, Any], redownload: bool) -> Path:
    sample_id = str(sample["id"])
    existing = video_path_for(video_dir, sample_id)
    if existing and not redownload:
        return existing

    video_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(video_dir / f"{sample_id}.%(ext)s")
    command = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--force-ipv4",
        "-f",
        "worst[ext=mp4]/worst",
        "-o",
        output_template,
        str(sample["url"]),
    ]
    run_command(command, root)
    path = video_path_for(video_dir, sample_id)
    if path is None:
        raise FileNotFoundError(f"No downloaded video found for {sample_id}")
    return path


def extract_frame(root: Path, ffmpeg: str, video: Path, output: Path, second: float) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-ss",
        str(second),
        "-i",
        str(video),
        "-frames:v",
        "1",
        "-update",
        "1",
        str(output),
    ]
    run_command(command, root)


def normalize_line(text: str) -> str:
    text = text.strip()
    text = SPEAKER_RE.sub("", text)
    text = text.replace("·", "，").replace("．", "。")
    text = re.sub(r"\s+", "", text)
    return text.strip()


def should_keep(text: str, confidence: float, min_confidence: float) -> bool:
    if confidence < min_confidence:
        return False
    if any(pattern in text for pattern in CREDIT_PATTERNS):
        return False
    if len(re.sub(r"[^\u4e00-\u9fa5]", "", text)) < 2:
        return False
    return True


def line_sort_key(item: Any) -> tuple[float, float]:
    box = item[0]
    xs = [point[0] for point in box]
    ys = [point[1] for point in box]
    return (sum(ys) / len(ys), sum(xs) / len(xs))


def ocr_frame(ocr: Any, image_path: Path, min_confidence: float) -> list[dict[str, Any]]:
    result, _ = ocr(str(image_path))
    rows: list[dict[str, Any]] = []
    for item in sorted(result or [], key=line_sort_key):
        raw = str(item[1])
        confidence = float(item[2])
        if not should_keep(raw, confidence, min_confidence):
            continue
        text = normalize_line(raw)
        if not text:
            continue
        rows.append({"text": text, "confidence": round(confidence, 4)})
    return rows


def dedupe_lines(lines: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for line in lines:
        if line in seen:
            continue
        seen.add(line)
        output.append(line)
    return output


def sample_times(sample: dict[str, Any], step: float) -> list[float]:
    start = float(sample.get("start", 0))
    duration = float(sample.get("duration", 20))
    times: list[float] = []
    current = start
    stop = start + duration
    while current <= stop:
        times.append(round(current, 3))
        current += step
    return times


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError as exc:
        raise RuntimeError("Install OCR deps first: .\\scripts\\install_deps.ps1 -Group ocr") from exc

    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = Path(args.manifest)
    output = Path(args.output)
    video_dir = Path(args.video_dir)
    frame_dir = Path(args.frame_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame_dir.mkdir(parents=True, exist_ok=True)

    ffmpeg = find_ffmpeg()
    ocr = RapidOCR()
    rows: list[dict[str, Any]] = []

    for sample in read_jsonl(manifest):
        sample_id = str(sample["id"])
        video = download_video(root, video_dir, sample, redownload=args.redownload)
        frame_rows: list[dict[str, Any]] = []
        lines: list[str] = []
        for second in sample_times(sample, args.frame_step):
            image_path = frame_dir / sample_id / f"{second:07.3f}.jpg"
            extract_frame(root, ffmpeg, video, image_path, second)
            ocr_lines = ocr_frame(ocr, image_path, args.min_confidence)
            frame_rows.append({"time": second, "image": str(image_path), "lines": ocr_lines})
            lines.extend(item["text"] for item in ocr_lines)

        reference = " ".join(dedupe_lines(lines))
        rows.append(
            {
                "id": sample_id,
                "title": sample.get("title"),
                "url": sample.get("url"),
                "start": sample.get("start", 0),
                "duration": sample.get("duration", 0),
                "ocr_reference": reference,
                "manual_reference": sample.get("reference"),
                "frames": frame_rows,
            }
        )
        print(f"{sample_id}: {reference}")

    with output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote OCR candidates to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
