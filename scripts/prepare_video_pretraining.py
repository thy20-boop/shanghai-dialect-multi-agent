from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".flv", ".webm"}
SUBTITLE_SUFFIXES = (".srt", ".vtt")


def find_ffmpeg(root: Path) -> tuple[str, str]:
    bundled = root / "external" / "ffmpeg-shared" / "bin"
    ffmpeg = bundled / "ffmpeg.exe"
    ffprobe = bundled / "ffprobe.exe"
    if ffmpeg.exists() and ffprobe.exists():
        return str(ffmpeg), str(ffprobe)
    return "ffmpeg", "ffprobe"


def run(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
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
    return completed


def ffprobe_json(ffprobe: str, path: Path, root: Path) -> dict:
    completed = run(
        [
            ffprobe,
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ],
        root,
    )
    return json.loads(completed.stdout)


def duration_seconds(metadata: dict) -> float:
    value = metadata.get("format", {}).get("duration")
    try:
        return round(float(value), 3)
    except (TypeError, ValueError):
        return 0.0


def subtitle_stream_count(metadata: dict) -> int:
    return sum(1 for stream in metadata.get("streams", []) if stream.get("codec_type") == "subtitle")


def find_sidecar(video: Path) -> Path | None:
    for suffix in SUBTITLE_SUFFIXES:
        candidate = video.with_suffix(suffix)
        if candidate.exists():
            return candidate
    return None


def extract_audio(ffmpeg: str, video: Path, output: Path, root: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
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


def parse_timestamp(value: str) -> float:
    value = value.strip().replace(",", ".")
    parts = value.split(":")
    if len(parts) == 3:
        hours, minutes, seconds = parts
    elif len(parts) == 2:
        hours = "0"
        minutes, seconds = parts
    else:
        return 0.0
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def clean_subtitle_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\{\\.*?\}", "", text)
    text = re.sub(r"\s+", "", text)
    return text.strip()


def parse_srt_or_vtt(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    blocks = re.split(r"\n\s*\n", text.replace("\r\n", "\n").replace("\r", "\n"))
    cues: list[dict] = []
    for block in blocks:
        lines = [line.strip() for line in block.split("\n") if line.strip()]
        if not lines:
            continue
        timing_index = next((index for index, line in enumerate(lines) if "-->" in line), None)
        if timing_index is None:
            continue
        timing = lines[timing_index]
        start_raw, end_raw = [part.strip().split()[0] for part in timing.split("-->", 1)]
        cue_text = clean_subtitle_text("".join(lines[timing_index + 1 :]))
        if not cue_text:
            continue
        cues.append({"start": parse_timestamp(start_raw), "end": parse_timestamp(end_raw), "text": cue_text})
    return cues


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare local Shanghainese videos for ASR/TTS pretraining.")
    parser.add_argument("--video-dir", default="data/raw_videos")
    parser.add_argument("--output-dir", default="outputs/video_pretraining")
    parser.add_argument("--skip-audio", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    video_dir = root / args.video_dir
    output_dir = root / args.output_dir
    audio_dir = output_dir / "audio_16k"
    output_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg, ffprobe = find_ffmpeg(root)

    videos = sorted(path for path in video_dir.iterdir() if path.suffix.lower() in VIDEO_SUFFIXES)
    sources: list[dict] = []
    asr_rows: list[dict] = []
    missing_subtitles: list[str] = []

    for index, video in enumerate(videos, start=1):
        sample_id = f"rawvideo_{index:04d}"
        metadata = ffprobe_json(ffprobe, video, root)
        audio_path = audio_dir / f"{sample_id}.wav"
        if not args.skip_audio:
            extract_audio(ffmpeg, video, audio_path, root)
        sidecar = find_sidecar(video)
        subtitle_cues = parse_srt_or_vtt(sidecar) if sidecar else []
        if not subtitle_cues:
            missing_subtitles.append(video.name)
        for cue_index, cue in enumerate(subtitle_cues, start=1):
            asr_rows.append(
                {
                    "id": f"{sample_id}_{cue_index:04d}",
                    "audio": str(audio_path),
                    "text": cue["text"],
                    "start": cue["start"],
                    "end": cue["end"],
                    "source_video": str(video),
                }
            )
        sources.append(
            {
                "id": sample_id,
                "video": str(video),
                "audio": str(audio_path),
                "duration": duration_seconds(metadata),
                "subtitle_streams": subtitle_stream_count(metadata),
                "sidecar_subtitle": str(sidecar) if sidecar else None,
                "subtitle_cues": len(subtitle_cues),
            }
        )
        print(f"{sample_id}: {video.name} duration={sources[-1]['duration']} cues={len(subtitle_cues)}")

    write_jsonl(output_dir / "video_sources.jsonl", sources)
    write_jsonl(output_dir / "asr_manifest_from_sidecars.jsonl", asr_rows)
    summary = {
        "video_count": len(videos),
        "audio_dir": str(audio_dir),
        "asr_rows": len(asr_rows),
        "missing_subtitle_count": len(missing_subtitles),
        "missing_subtitles": missing_subtitles,
        "next_step": "Add sidecar subtitles or run hard-subtitle OCR to create labeled ASR/TTS training examples.",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
