from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any


def run(command: list[str], cwd: Path) -> None:
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed ({completed.returncode}): {' '.join(command)}\n{completed.stdout}")


def find_ffmpeg(root: Path) -> str:
    bundled = root / "external" / "ffmpeg-shared" / "bin" / "ffmpeg.exe"
    return str(bundled) if bundled.exists() else "ffmpeg"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_frame(ffmpeg: str, video: Path, second: float, output: Path, root: Path, crop_bottom: float) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    vf = f"crop=iw:ih*{crop_bottom}:0:ih*(1-{crop_bottom})"
    run(
        [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{second:.3f}",
            "-i",
            str(video),
            "-vf",
            vf,
            "-frames:v",
            "1",
            "-update",
            "1",
            str(output),
        ],
        root,
    )


def try_extract_frame(ffmpeg: str, video: Path, second: float, output: Path, root: Path, crop_bottom: float) -> bool:
    try:
        extract_frame(ffmpeg, video, second, output, root, crop_bottom=crop_bottom)
    except RuntimeError:
        return False
    return output.exists() and output.stat().st_size > 0


def extract_clip(ffmpeg: str, video: Path, start: float, end: float, output: Path, root: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.2, end - start)
    run(
        [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{max(0.0, start):.3f}",
            "-t",
            f"{duration:.3f}",
            "-i",
            str(video),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-sample_fmt",
            "s16",
            str(output),
        ],
        root,
    )


def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", "", text.strip())
    text = text.replace("，", "").replace("。", "").replace("？", "").replace("?", "")
    text = text.replace("：", "").replace(":", "")
    return text


def useful_text(text: str) -> bool:
    chinese = re.sub(r"[^\u4e00-\u9fff]", "", text)
    if len(chinese) < 3:
        return False
    if "bilibili" in text.lower() or "哔哩" in text:
        return False
    if "上海话情景对话" in text:
        return False
    return True


def ocr_lines(ocr: Any, image: Path, min_confidence: float) -> list[dict[str, Any]]:
    result, _ = ocr(str(image))
    rows: list[dict[str, Any]] = []
    for item in result or []:
        box = item[0]
        text = normalize_text(str(item[1]))
        confidence = float(item[2])
        if confidence < min_confidence or not useful_text(text):
            continue
        ys = [point[1] for point in box]
        xs = [point[0] for point in box]
        rows.append(
            {
                "text": text,
                "confidence": confidence,
                "y": sum(ys) / len(ys),
                "x": sum(xs) / len(xs),
            }
        )
    rows.sort(key=lambda row: (row["y"], row["x"]))
    return rows


def cue_from_lines(lines: list[dict[str, Any]]) -> tuple[str, str] | None:
    if not lines:
        return None
    if len(lines) == 1:
        return lines[0]["text"], lines[0]["text"]
    # In these videos the upper line is Shanghainese/Wu script and the lower line
    # is Mandarin. Keep both so we can train either task later.
    return lines[0]["text"], lines[-1]["text"]


def merge_frame_cues(frame_cues: list[dict[str, Any]], step: float, min_duration: float) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for item in frame_cues:
        key = (item["wu_text"], item["mandarin"])
        if current and key == (current["wu_text"], current["mandarin"]):
            current["end"] = item["time"] + step
            current["frames"] += 1
            continue
        if current and current["end"] - current["start"] >= min_duration:
            merged.append(current)
        current = {
            "source_id": item["source_id"],
            "source_video": item["source_video"],
            "start": item["time"],
            "end": item["time"] + step,
            "wu_text": item["wu_text"],
            "mandarin": item["mandarin"],
            "frames": 1,
        }
    if current and current["end"] - current["start"] >= min_duration:
        merged.append(current)
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description="OCR hard subtitles from local videos and build clipped ASR manifests.")
    parser.add_argument("--sources", default="outputs/video_pretraining/video_sources.jsonl")
    parser.add_argument("--output-dir", default="outputs/video_pretraining/hardsub")
    parser.add_argument("--frame-step", type=float, default=1.0)
    parser.add_argument("--min-confidence", type=float, default=0.65)
    parser.add_argument("--min-duration", type=float, default=1.0)
    parser.add_argument("--crop-bottom", type=float, default=0.5, help="Only OCR the lower fraction of each frame.")
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--skip-clips", action="store_true")
    args = parser.parse_args()

    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError as exc:
        raise RuntimeError("Install OCR deps first: .\\scripts\\install_deps.ps1 -Group ocr") from exc

    root = Path(__file__).resolve().parents[1]
    output_dir = root / args.output_dir
    frame_dir = output_dir / "frames"
    clip_dir = output_dir / "clips_16k"
    ffmpeg = find_ffmpeg(root)
    ocr = RapidOCR()

    sources = read_jsonl(root / args.sources)
    if args.max_videos is not None:
        sources = sources[: args.max_videos]

    all_cues: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    for source in sources:
        source_id = source["id"]
        video = Path(source["video"])
        duration = float(source.get("duration") or 0.0)
        time = 0.0
        while time < max(0.0, duration - 0.25):
            image = frame_dir / source_id / f"{time:08.3f}.jpg"
            if not try_extract_frame(ffmpeg, video, time, image, root, crop_bottom=args.crop_bottom):
                time += args.frame_step
                continue
            pair = cue_from_lines(ocr_lines(ocr, image, args.min_confidence))
            if pair:
                wu_text, mandarin = pair
                frame_rows.append(
                    {
                        "source_id": source_id,
                        "source_video": str(video),
                        "time": round(time, 3),
                        "wu_text": wu_text,
                        "mandarin": mandarin,
                    }
                )
            time += args.frame_step
        source_cues = merge_frame_cues(
            [row for row in frame_rows if row["source_id"] == source_id],
            step=args.frame_step,
            min_duration=args.min_duration,
        )
        all_cues.extend(source_cues)
        print(f"{source_id}: frame_rows={sum(1 for row in frame_rows if row['source_id'] == source_id)} cues={len(source_cues)}")

    if not args.skip_clips:
        for index, cue in enumerate(all_cues, start=1):
            clip = clip_dir / f"cue_{index:05d}.wav"
            extract_clip(ffmpeg, Path(cue["source_video"]), float(cue["start"]), float(cue["end"]), clip, root)
            cue["audio"] = str(clip)

    write_jsonl(output_dir / "frame_ocr.jsonl", frame_rows)
    write_jsonl(output_dir / "cues.jsonl", all_cues)
    write_jsonl(
        output_dir / "asr_wu_manifest.jsonl",
        [{"audio": row.get("audio"), "text": row["wu_text"]} for row in all_cues if row.get("audio")],
    )
    write_jsonl(
        output_dir / "asr_mandarin_manifest.jsonl",
        [{"audio": row.get("audio"), "text": row["mandarin"]} for row in all_cues if row.get("audio")],
    )
    summary = {
        "source_videos": len(sources),
        "frame_rows": len(frame_rows),
        "cues": len(all_cues),
        "output_dir": str(output_dir),
        "wu_manifest": str(output_dir / "asr_wu_manifest.jsonl"),
        "mandarin_manifest": str(output_dir / "asr_mandarin_manifest.jsonl"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
