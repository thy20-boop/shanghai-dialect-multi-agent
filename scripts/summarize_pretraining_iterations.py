from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize ASR pretraining iteration metrics.")
    parser.add_argument("--models-root", default="outputs/models")
    parser.add_argument("--output", default="outputs/video_pretraining/pretrain_iterations.md")
    args = parser.parse_args()

    models_root = Path(args.models_root)
    candidates = sorted(models_root.glob("whisper-small-shanghai-direct-video-pretrain*"))
    rows = []
    for model_dir in candidates:
        metadata_path = model_dir / "training_metadata.json"
        if not metadata_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        rows.append(render_row(model_dir, metadata))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_markdown(rows), encoding="utf-8")
    print(f"Wrote {output}")
    return 0


def render_row(model_dir: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    final_eval = metadata.get("final_eval") or {}
    return {
        "name": model_dir.name,
        "path": str(model_dir),
        "lr": metadata.get("learning_rate"),
        "max_steps": metadata.get("max_steps", -1),
        "epochs": metadata.get("num_train_epochs"),
        "train_examples": metadata.get("train_examples"),
        "eval_examples": metadata.get("eval_examples"),
        "best_cer": metadata.get("best_metric"),
        "best_checkpoint": metadata.get("best_model_checkpoint"),
        "final_cer": final_eval.get("eval_cer"),
        "final_loss": final_eval.get("eval_loss"),
    }


def render_markdown(rows: list[dict[str, Any]]) -> str:
    rows = sorted(rows, key=lambda row: (row["best_cer"] is None, row["best_cer"] or 999))
    lines = [
        "# ASR Pretraining Iterations",
        "",
        "| version | learning rate | max steps | best CER | final CER | best checkpoint |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {name} | {lr} | {max_steps} | {best_cer} | {final_cer} | `{best_checkpoint}` |".format(
                name=row["name"],
                lr=row["lr"],
                max_steps=row["max_steps"],
                best_cer=format_float(row["best_cer"]),
                final_cer=format_float(row["final_cer"]),
                best_checkpoint=row["best_checkpoint"],
            )
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- Lower CER is better.",
            "- The final saved adapter uses the best checkpoint because training loads the best model at the end.",
            "- The validation manifest is the combined direct-Mandarin dev split built from the original 92 examples plus local hard-subtitle video clips.",
            "",
        ]
    )
    return "\n".join(lines)


def format_float(value: Any) -> str:
    if isinstance(value, int | float):
        return f"{value:.4f}"
    return ""


if __name__ == "__main__":
    raise SystemExit(main())
