from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class ActiveLearningSummary:
    total: int
    pending: int
    confirmed: int
    exported_ready: int
    unique_audio: int
    reason_counts: dict[str, int]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def append_active_learning_items(path: str | Path | None, items: list[dict[str, Any]]) -> int:
    """Append active-learning candidates to a JSONL queue with de-duplication."""

    if not path or not items:
        return 0
    queue_path = Path(path)
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    existing_ids = _read_existing_ids(queue_path)
    records: list[dict[str, Any]] = []
    for item in items:
        item_id = active_learning_item_id(item)
        if item_id in existing_ids:
            continue
        record = {
            "id": item_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "needs_human_review",
            **item,
        }
        records.append(record)
        existing_ids.add(item_id)

    if not records:
        return 0
    with queue_path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(records)


def read_active_learning_queue(path: str | Path | None) -> list[dict[str, Any]]:
    if not path:
        return []
    queue_path = Path(path)
    if not queue_path.exists():
        return []
    records: list[dict[str, Any]] = []
    with queue_path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                records.append(row)
    return records


def summarize_active_learning_items(items: list[dict[str, Any]]) -> ActiveLearningSummary:
    reason_counts: Counter[str] = Counter()
    audio_paths: set[str] = set()
    pending = 0
    confirmed = 0
    exported_ready = 0
    for item in items:
        status = str(item.get("status") or "needs_human_review")
        if status in {"confirmed", "approved"}:
            confirmed += 1
        elif status not in {"exported", "ignored"}:
            pending += 1

        audio_path = item.get("audio_path") or item.get("audio")
        if audio_path:
            audio_paths.add(str(audio_path))
        for reason in _normalize_reasons(item.get("reason")):
            reason_counts[reason] += 1
        if _confirmed_text(item):
            exported_ready += 1

    return ActiveLearningSummary(
        total=len(items),
        pending=pending,
        confirmed=confirmed,
        exported_ready=exported_ready,
        unique_audio=len(audio_paths),
        reason_counts=dict(sorted(reason_counts.items())),
    )


def render_active_learning_report(
    items: list[dict[str, Any]],
    title: str = "Active Learning Queue Report",
) -> str:
    summary = summarize_active_learning_items(items)
    lines = [
        f"# {title}",
        "",
        "## Summary",
        "",
        f"- Total items: {summary.total}",
        f"- Pending review: {summary.pending}",
        f"- Confirmed: {summary.confirmed}",
        f"- Export-ready: {summary.exported_ready}",
        f"- Unique audio files: {summary.unique_audio}",
        "",
        "## Reasons",
        "",
    ]
    if summary.reason_counts:
        for reason, count in summary.reason_counts.items():
            lines.append(f"- `{reason}`: {count}")
    else:
        lines.append("- none")

    lines.extend(["", "## Items", ""])
    if not items:
        lines.append("- none")
    for item in items:
        item_id = item.get("id", "unknown")
        status = item.get("status", "needs_human_review")
        reasons = ", ".join(_normalize_reasons(item.get("reason"))) or "unknown"
        transcript = str(item.get("primary_transcript") or "").strip()
        confirmed_text = _confirmed_text(item)
        lines.append(f"- `{item_id}` `{status}` reasons={reasons}")
        if transcript:
            lines.append(f"  - primary: {transcript}")
        if confirmed_text:
            lines.append(f"  - confirmed: {confirmed_text}")
    return "\n".join(lines) + "\n"


def export_active_learning_manifest(
    queue_path: str | Path | None,
    output_path: str | Path,
    include_unconfirmed: bool = False,
) -> int:
    items = read_active_learning_queue(queue_path)
    rows: list[dict[str, Any]] = []
    for item in items:
        audio = item.get("audio_path") or item.get("audio")
        if not audio:
            continue
        text = _confirmed_text(item)
        if not text and include_unconfirmed:
            text = str(item.get("primary_transcript") or "").strip()
        if not text:
            continue
        rows.append(
            {
                "id": item.get("id") or active_learning_item_id(item),
                "audio": audio,
                "text": text,
                "source": "active_learning",
                "status": item.get("status", "needs_human_review"),
                "reason": _normalize_reasons(item.get("reason")),
            }
        )

    manifest_path = Path(output_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def active_learning_item_id(item: dict[str, Any]) -> str:
    payload = {
        "audio_path": item.get("audio_path"),
        "reason": item.get("reason"),
        "primary_transcript": item.get("primary_transcript"),
        "candidate_transcripts": item.get("candidate_transcripts"),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _read_existing_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    for row in read_active_learning_queue(path):
        item_id = row.get("id")
        if item_id:
            ids.add(str(item_id))
    return ids


def _normalize_reasons(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)]


def _confirmed_text(item: dict[str, Any]) -> str:
    for key in ("confirmed_transcript", "confirmed_text", "reference", "text"):
        value = str(item.get(key) or "").strip()
        if value:
            return value
    return ""
