from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ganagent.agent import ShanghaiDialectAgent
from ganagent.asr_backends import make_backend
from ganagent.dashboard import render_html_dashboard
from ganagent.evaluation import evaluate_pairs
from ganagent.io import read_jsonl, write_jsonl
from ganagent.report import render_markdown_report
from ganagent.repair import RepairEngine, count_repair_actions


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Shanghai dialect ASR agent pipeline.")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--mode", choices=["mock", "real"], default="mock")
    parser.add_argument("--manifest", default=None, help="Override manifest path.")
    parser.add_argument("--backend", default=None, choices=["mock", "whisper"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    config = load_config(ROOT / args.config)
    outputs = config["outputs"]
    output_dir = ROOT / outputs["dir"]
    reports_dir = ROOT / outputs["reports_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    manifest = choose_manifest(config, args.mode, args.manifest)
    backend_name = args.backend or ("mock" if args.mode == "mock" else config["asr"]["backend"])
    model = args.model or config["asr"].get("model")
    local_files_only = args.local_files_only or bool(config["asr"].get("local_files_only", False))

    repair_engine = RepairEngine.from_file(ROOT / config["glossary"])
    backend = make_backend(backend_name, model_name=model, local_files_only=local_files_only)
    agent = ShanghaiDialectAgent(asr_backend=backend, repair_engine=repair_engine)

    predictions = run_predictions(agent, ROOT / manifest, reports_dir)
    predictions_path = ROOT / outputs["predictions"]
    write_jsonl(predictions_path, predictions)

    summary = evaluate_predictions(predictions, repair_engine)
    evaluation_path = ROOT / outputs["evaluation"]
    evaluation_path.parent.mkdir(parents=True, exist_ok=True)
    evaluation_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    dashboard_path = ROOT / outputs["dashboard"]
    dashboard_path.write_text(
        render_html_dashboard(predictions, summary, title=config["project_name"]),
        encoding="utf-8",
    )

    print(f"Manifest: {manifest}")
    print(f"Predictions: {predictions_path}")
    print(f"Evaluation: {evaluation_path}")
    print(f"Dashboard: {dashboard_path}")
    return 0


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def choose_manifest(config: dict[str, Any], mode: str, override: str | None) -> str:
    if override:
        return override
    if mode == "mock":
        return config["mock_manifest"]
    return config["dataset"]["manifest"]


def run_predictions(
    agent: ShanghaiDialectAgent,
    manifest_path: Path,
    reports_dir: Path,
) -> list[dict[str, Any]]:
    rows = read_jsonl(manifest_path)
    outputs: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        audio = row.get("audio")
        result = agent.run(audio_path=str(audio) if audio else None)
        output = {
            "id": row.get("id", index),
            "audio": audio,
            "reference": row.get("reference") or row.get("text"),
            "transcript": result.transcript,
            "mandarin_translation": result.mandarin_translation,
            "dialect": result.dialect.label,
            "dialect_score": result.dialect.score,
            "dialect_markers": result.dialect.markers,
            "repair_count": count_repair_actions(result.repairs),
            "suspicion_count": len(result.suspicions),
            "repairs": result.repairs,
            "suspicions": [item.__dict__ for item in result.suspicions],
        }
        outputs.append(output)
        (reports_dir / f"sample_{index:04d}.md").write_text(
            render_markdown_report(result),
            encoding="utf-8",
        )
    return outputs


def evaluate_predictions(predictions: list[dict[str, Any]], repair_engine: RepairEngine) -> dict[str, Any]:
    pairs = [
        (str(row["reference"]), str(row["transcript"]))
        for row in predictions
        if row.get("reference") is not None and row.get("transcript") is not None
    ]
    return evaluate_pairs(
        pairs,
        domain_terms=repair_engine.glossary.get("domain_terms", []),
        dialect_markers=repair_engine.glossary.get("dialect_terms", {}).keys(),
    ).as_dict()


if __name__ == "__main__":
    raise SystemExit(main())
