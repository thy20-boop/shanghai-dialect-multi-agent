from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
import shutil
from typing import Any


DEFAULT_MANIFESTS = (
    Path("data/splits/train.jsonl"),
    Path("data/splits/dev.jsonl"),
)


@dataclass
class VoiceCloneItem:
    audio: str
    text: str
    speaker_id: str
    gender: str | None = None
    split: str | None = None
    duration: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SpeakerStats:
    speaker_id: str
    gender: str | None
    count: int
    duration: float
    chars: int

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["duration_minutes"] = round(self.duration / 60, 3)
        return payload


@dataclass
class VoiceCloneExport:
    output_dir: str
    speaker_id: str
    item_count: int
    duration_minutes: float
    metadata_csv: str
    gpt_sovits_list: str
    cosyvoice_manifest: str
    reference_audio: str
    reference_text: str
    speaker_stats: list[dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_voice_clone_items(
    root: str | Path = ".",
    manifests: tuple[Path, ...] = DEFAULT_MANIFESTS,
) -> list[VoiceCloneItem]:
    root_path = Path(root)
    items: list[VoiceCloneItem] = []
    for manifest in manifests:
        manifest_path = root_path / manifest
        if not manifest_path.exists():
            continue
        split = manifest_path.stem
        with manifest_path.open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                audio = str(row.get("audio") or "").strip()
                text = str(row.get("text") or "").strip()
                speaker_id = str(row.get("speaker_id") or "unknown").strip()
                if not audio or not text:
                    continue
                items.append(
                    VoiceCloneItem(
                        audio=audio,
                        text=text,
                        speaker_id=speaker_id,
                        gender=row.get("gender"),
                        split=split,
                        duration=audio_duration_seconds(root_path / audio),
                    )
                )
    return items


def audio_duration_seconds(path: str | Path) -> float:
    try:
        import soundfile as sf
    except ImportError:
        return 0.0
    try:
        info = sf.info(str(path))
    except Exception:
        return 0.0
    if not info.samplerate:
        return 0.0
    return float(info.frames) / float(info.samplerate)


def speaker_statistics(items: list[VoiceCloneItem]) -> list[SpeakerStats]:
    grouped: dict[str, dict[str, Any]] = {}
    for item in items:
        slot = grouped.setdefault(
            item.speaker_id,
            {"gender": item.gender, "count": 0, "duration": 0.0, "chars": 0},
        )
        slot["count"] += 1
        slot["duration"] += item.duration
        slot["chars"] += len(item.text)
    stats = [
        SpeakerStats(
            speaker_id=speaker_id,
            gender=data["gender"],
            count=data["count"],
            duration=data["duration"],
            chars=data["chars"],
        )
        for speaker_id, data in grouped.items()
    ]
    return sorted(stats, key=lambda item: (item.duration, item.count), reverse=True)


def choose_speaker(items: list[VoiceCloneItem], requested: str = "auto") -> str:
    if requested and requested != "auto":
        return requested
    stats = speaker_statistics(items)
    if not stats:
        raise ValueError("No speaker data found for voice cloning.")
    return stats[0].speaker_id


def export_voice_clone_assets(
    output_dir: str | Path,
    root: str | Path = ".",
    speaker_id: str = "auto",
    max_items: int = 240,
    max_reference_clips: int = 5,
    min_duration: float = 1.0,
    max_duration: float = 12.0,
    clean_only: bool = False,
    max_per_speaker: int | None = None,
) -> VoiceCloneExport:
    root_path = Path(root)
    output_path = Path(output_dir)
    items = load_voice_clone_items(root_path)
    use_all_speakers = speaker_id.strip().lower() in {"all", "multi", "multi_speaker", "multispeaker"}
    selected_speaker = "all" if use_all_speakers else choose_speaker(items, speaker_id)
    speaker_items = [
        item
        for item in items
        if (use_all_speakers or item.speaker_id == selected_speaker) and min_duration <= item.duration <= max_duration
    ]
    if clean_only:
        speaker_items = [item for item in speaker_items if is_clean_tts_item(item)]
    speaker_items = select_tts_training_items(
        speaker_items,
        max_items=max_items,
        balance_speakers=use_all_speakers,
        max_per_speaker=max_per_speaker,
    )
    if not speaker_items:
        raise ValueError(f"No usable clips found for speaker {selected_speaker}.")
    export_items = speaker_items[:max_items]

    audio_dir = output_path / "audio"
    ref_dir = output_path / "reference"
    audio_dir.mkdir(parents=True, exist_ok=True)
    ref_dir.mkdir(parents=True, exist_ok=True)

    copied_items: list[VoiceCloneItem] = []
    for index, item in enumerate(export_items, start=1):
        source = root_path / item.audio
        target = audio_dir / f"{item.speaker_id}_{index:04d}{source.suffix.lower() or '.wav'}"
        shutil.copy2(source, target)
        copied_items.append(
            VoiceCloneItem(
                audio=str(target),
                text=item.text,
                speaker_id=item.speaker_id,
                gender=item.gender,
                split=item.split,
                duration=item.duration,
            )
        )

    reference_pool = [item for item in export_items if item.speaker_id == choose_speaker(items, "auto")] or export_items
    reference_items = sorted(reference_pool, key=lambda item: (abs(item.duration - 6.0), len(item.text)))[:max_reference_clips]
    reference_paths: list[Path] = []
    for index, item in enumerate(reference_items, start=1):
        source = root_path / item.audio
        target = ref_dir / f"{item.speaker_id}_ref_{index:02d}{source.suffix.lower() or '.wav'}"
        shutil.copy2(source, target)
        reference_paths.append(target)

    metadata_csv = output_path / "metadata.csv"
    with metadata_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, delimiter="|")
        writer.writerow(["audio", "text", "speaker_id", "duration", "split"])
        for item in copied_items:
            writer.writerow([item.audio, item.text, item.speaker_id, f"{item.duration:.3f}", item.split or ""])

    gpt_sovits_list = output_path / "gpt_sovits.list"
    with gpt_sovits_list.open("w", encoding="utf-8") as handle:
        for item in copied_items:
            handle.write(f"{Path(item.audio).resolve()}|{item.speaker_id}|zh|{item.text}\n")

    cosyvoice_manifest = output_path / "cosyvoice_manifest.jsonl"
    with cosyvoice_manifest.open("w", encoding="utf-8") as handle:
        for item in copied_items:
            handle.write(
                json.dumps(
                    {
                        "audio": str(Path(item.audio).resolve()),
                        "text": item.text,
                        "speaker_id": item.speaker_id,
                        "duration": round(item.duration, 3),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    reference_text = output_path / "reference_prompt.txt"
    reference_text.write_text(reference_items[0].text, encoding="utf-8")

    readme = output_path / "README.md"
    readme.write_text(
        render_voice_clone_readme(
            selected_speaker=selected_speaker,
            item_count=len(copied_items),
            duration=sum(item.duration for item in copied_items),
            reference_audio=reference_paths[0],
            reference_text=reference_items[0].text,
        ),
        encoding="utf-8",
    )

    stats_path = output_path / "speaker_stats.json"
    stats_payload = [stat.as_dict() for stat in speaker_statistics(items)]
    stats_path.write_text(json.dumps(stats_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return VoiceCloneExport(
        output_dir=str(output_path),
        speaker_id=selected_speaker,
        item_count=len(copied_items),
        duration_minutes=round(sum(item.duration for item in copied_items) / 60, 3),
        metadata_csv=str(metadata_csv),
        gpt_sovits_list=str(gpt_sovits_list),
        cosyvoice_manifest=str(cosyvoice_manifest),
        reference_audio=str(reference_paths[0]),
        reference_text=str(reference_text),
        speaker_stats=[stat.as_dict() for stat in speaker_statistics(items)[:10]],
    )


def is_clean_tts_item(item: VoiceCloneItem) -> bool:
    text = item.text.strip()
    if not 8 <= len(text) <= 70:
        return False
    if any(token in text for token in ["哈哈", "音乐", "字幕", "谢谢观看", "[", "]"]):
        return False
    filler_count = sum(text.count(token) for token in ["呃", "啊", "嗯", "诶"])
    if filler_count > 5:
        return False
    if filler_count / max(1, len(text)) > 0.14:
        return False
    if "呃呃" in text or "啊啊" in text or "嗯嗯" in text:
        return False
    return True


def tts_item_score(item: VoiceCloneItem) -> tuple[float, int]:
    text = item.text.strip()
    filler_count = sum(text.count(token) for token in ["呃", "啊", "嗯", "诶"])
    duration_penalty = abs(item.duration - 5.0)
    length_penalty = abs(len(text) - 30) / 20
    punctuation_penalty = text.count("，") * 0.2 + text.count("、") * 0.2
    return (filler_count * 2 + duration_penalty + length_penalty + punctuation_penalty, len(text))


def select_tts_training_items(
    items: list[VoiceCloneItem],
    max_items: int,
    balance_speakers: bool,
    max_per_speaker: int | None = None,
) -> list[VoiceCloneItem]:
    if not balance_speakers:
        return sorted(items, key=tts_item_score)

    grouped: dict[str, list[VoiceCloneItem]] = {}
    for item in items:
        grouped.setdefault(item.speaker_id, []).append(item)
    for speaker_items in grouped.values():
        speaker_items.sort(key=tts_item_score)

    selected: list[VoiceCloneItem] = []
    counts = {speaker_id: 0 for speaker_id in grouped}
    while len(selected) < max_items:
        progressed = False
        for speaker_id in sorted(grouped):
            if len(selected) >= max_items:
                break
            if max_per_speaker is not None and counts[speaker_id] >= max_per_speaker:
                continue
            speaker_items = grouped[speaker_id]
            if not speaker_items:
                continue
            selected.append(speaker_items.pop(0))
            counts[speaker_id] += 1
            progressed = True
        if not progressed:
            break
    return selected


def render_voice_clone_readme(
    selected_speaker: str,
    item_count: int,
    duration: float,
    reference_audio: Path,
    reference_text: str,
) -> str:
    duration_minutes = round(duration / 60, 3)
    return f"""# 吴语/上海话 Voice Cloning 数据包

说话人：`{selected_speaker}`

导出样本：`{item_count}` 条，约 `{duration_minutes}` 分钟。

## 文件

- `metadata.csv`：通用 TTS 训练清单。
- `gpt_sovits.list`：GPT-SoVITS 训练/切分常用格式，字段为 `wav|speaker|lang|text`。
- `cosyvoice_manifest.jsonl`：CosyVoice 或自定义训练脚本可读取的 JSONL。
- `reference/`：zero-shot voice cloning 参考音频。
- `reference_prompt.txt`：第一条参考音频对应的上海话转写文本。
- `speaker_stats.json`：训练集说话人统计。

## 推荐 zero-shot 调用

参考音频：

```text
{reference_audio}
```

参考文本：

```text
{reference_text}
```

如果使用 GPT-SoVITS API，可启动其 `api_v2.py` 后在本项目中运行：

```powershell
.\\.venv\\Scripts\\python.exe -m ganagent.cli speak --target wuu --text "侬好，阿拉来试试看。" --output outputs\\wu_clone.wav --tts-backend gpt_sovits --ref-audio "{reference_audio}" --prompt-text "{reference_text}"
```

注意：训练集是自然对话 ASR 数据，不是录音棚 TTS 数据。它适合做课程展示和 few-shot/zero-shot 克隆参考，但如果要高质量商用 TTS，还需要更干净、更长、更均衡的单说话人语料。
"""
