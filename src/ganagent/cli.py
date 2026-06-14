from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from ganagent.agent import ShanghaiDialectAgent
from ganagent.asr_backends import make_backend
from ganagent.codex_task import render_codex_answer_task, render_codex_task_metadata
from ganagent.evaluation import evaluate_pairs
from ganagent.io import read_jsonl, write_jsonl
from ganagent.learning import (
    append_active_learning_items,
    export_active_learning_manifest,
    read_active_learning_queue,
    render_active_learning_report,
    summarize_active_learning_items,
)
from ganagent.product import build_translation_product
from ganagent.report import render_markdown_report
from ganagent.speech_quality import score_spoken_answer
from ganagent.repair import (
    RepairEngine,
    count_repair_actions,
    load_custom_repairs_file,
    parse_custom_repairs,
)
from ganagent.tts import (
    DEFAULT_COSYVOICE_WU_URL,
    DEFAULT_GPT_SOVITS_URL,
    DEFAULT_TTS_BACKEND,
    DEFAULT_TTS_VOICE,
    DEFAULT_WU_TTS_BACKEND,
    DEFAULT_WU_TTS_VOICE,
    TTSRequest,
    allows_prefix_trim,
    join_wav_files,
    leading_hallucination_chars,
    load_wu_reference_experts,
    resolve_tts_text,
    select_wu_reference_experts,
    split_tts_sentences,
    synthesize_mp3,
    trim_wav_leading_hallucination,
    wu_voice_notice,
    WuReferenceExpert,
)
from ganagent.voice_clone import export_voice_clone_assets


TRAINED_ASR_MODEL = Path("outputs/models/whisper-small-shanghai-lora-full")
BASE_ASR_MODEL = "TingChen-ppmc/whisper-small-Shanghai"
DEFAULT_ASR_MODEL = os.environ.get("SHANGHAI_ASR_MODEL") or (
    str(TRAINED_ASR_MODEL) if TRAINED_ASR_MODEL.exists() else BASE_ASR_MODEL
)
DEFAULT_ASR_BACKEND = os.environ.get("SHANGHAI_ASR_BACKEND", "dolphin_multiagent")
ASR_BACKENDS = [
    "mock",
    "whisper",
    "whisper_medium_wu",
    "funasr",
    "dolphin",
    "hybrid",
    "dolphin_multiagent",
]
ASSIST_BACKENDS = ["none", "whisper", "whisper_medium_wu", "funasr", "dolphin"]
DEFAULT_REPAIR_MEMORY = os.environ.get("SHANGHAI_REPAIR_MEMORY", "data/user_corrections.json")
DEFAULT_ACTIVE_LEARNING_LOG = os.environ.get(
    "SHANGHAI_ACTIVE_LEARNING_LOG",
    "data/active_learning_queue.jsonl",
)
DEFAULT_TTS_REVIEW_LOG = os.environ.get("SHANGHAI_TTS_REVIEW_LOG", "data/tts_quality_queue.jsonl")


def add_runtime_agent_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--memory", default=DEFAULT_REPAIR_MEMORY, help="Optional persistent repair memory JSON.")
    parser.add_argument(
        "--active-learning-log",
        default=DEFAULT_ACTIVE_LEARNING_LOG,
        help="JSONL queue for high-risk or disagreement samples.",
    )
    parser.add_argument(
        "--no-save-active-learning",
        action="store_true",
        help="Do not append active-learning candidates to the queue.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Shanghainese/Wu dialect ASR repair agent")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="Run a deterministic mock demo.")
    demo.add_argument("--glossary", default=None, help="Path to glossary JSON.")
    add_runtime_agent_arguments(demo)
    demo.add_argument("--json", action="store_true", help="Print raw JSON.")

    transcribe = subparsers.add_parser("transcribe", help="Run the agent on audio.")
    transcribe.add_argument("--audio", required=True, help="Audio file path.")
    transcribe.add_argument("--backend", default=DEFAULT_ASR_BACKEND, choices=ASR_BACKENDS)
    transcribe.add_argument("--model", default=None, help="ASR model name or local path.")
    transcribe.add_argument("--assist-backend", default="none", choices=ASSIST_BACKENDS)
    transcribe.add_argument("--assist-model", default=None, help="Optional open-source ASR candidate model.")
    transcribe.add_argument("--glossary", default=None, help="Path to glossary JSON.")
    add_runtime_agent_arguments(transcribe)
    transcribe.add_argument(
        "--custom-repair",
        action="append",
        default=[],
        help="Extra post-ASR repair in the form wrong=correct. Can be repeated.",
    )
    transcribe.add_argument("--local-files-only", action="store_true")
    transcribe.add_argument("--chunk-seconds", type=float, default=15.0, help="Whisper chunk size for long audio.")
    transcribe.add_argument("--no-vad", action="store_true", help="Disable pause-based long audio segmentation.")
    transcribe.add_argument(
        "--max-speech-region-seconds",
        type=float,
        default=8.0,
        help="Maximum speech segment duration when VAD segmentation is enabled.",
    )
    transcribe.add_argument("--json", action="store_true", help="Print raw JSON.")
    transcribe.add_argument("--markdown", default=None, help="Write a Markdown report.")

    translate = subparsers.add_parser("translate", help="Translate Shanghainese audio to Mandarin.")
    translate.add_argument("--audio", required=True, help="Audio file path.")
    translate.add_argument("--backend", default=DEFAULT_ASR_BACKEND, choices=ASR_BACKENDS)
    translate.add_argument(
        "--model",
        default=None,
        help="ASR model name or local path. Defaults to small.cn for dolphin_multiagent/dolphin and the trained LoRA for hybrid/whisper.",
    )
    translate.add_argument("--assist-backend", default="none", choices=ASSIST_BACKENDS)
    translate.add_argument("--assist-model", default=None, help="Optional open-source ASR candidate model.")
    translate.add_argument("--glossary", default=None, help="Path to glossary JSON.")
    add_runtime_agent_arguments(translate)
    translate.add_argument(
        "--custom-repair",
        action="append",
        default=[],
        help="Extra post-ASR repair in the form wrong=correct. Can be repeated.",
    )
    translate.add_argument("--local-files-only", action="store_true")
    translate.add_argument("--chunk-seconds", type=float, default=15.0, help="Whisper chunk size for long audio.")
    translate.add_argument("--no-vad", action="store_true", help="Disable pause-based long audio segmentation.")
    translate.add_argument(
        "--max-speech-region-seconds",
        type=float,
        default=8.0,
        help="Maximum speech segment duration when VAD segmentation is enabled.",
    )
    translate.add_argument("--json", action="store_true", help="Print raw JSON.")
    translate.add_argument("--markdown", default=None, help="Write a Markdown report.")
    translate.add_argument("--tts-output", default=None, help="Write Mandarin or Wu-style speech as an MP3 file.")
    translate.add_argument(
        "--tts-target",
        choices=["mandarin", "wuu"],
        default="mandarin",
        help="Speech content: Mandarin translation or Wu oral script.",
    )
    translate.add_argument("--tts-backend", choices=["edge", "cosyvoice_wu", "command"], default=None)
    translate.add_argument("--tts-voice", default=DEFAULT_TTS_VOICE, help="edge-tts voice name.")
    translate.add_argument("--tts-rate", default="+0%", help="edge-tts speaking rate, for example +0%% or -10%%.")
    translate.add_argument("--tts-pitch", default="+0Hz", help="edge-tts pitch, for example +0Hz.")
    translate.add_argument("--ref-audio", default=None, help="Reference audio for GPT-SoVITS or external voice cloning.")
    translate.add_argument("--prompt-text", default="", help="Transcript for the reference audio.")
    translate.add_argument("--text-lang", default="zh")
    translate.add_argument("--prompt-lang", default="zh")
    translate.add_argument("--gpt-sovits-url", default=DEFAULT_GPT_SOVITS_URL)
    translate.add_argument("--cosyvoice-wu-url", default=DEFAULT_COSYVOICE_WU_URL)
    translate.add_argument("--tts-command", default=None, help="External command template. Placeholders: {text_file}, {output}, {ref_audio}, {prompt_text}.")
    translate.add_argument(
        "--codex-task-output",
        default=None,
        help="Write a Markdown task for Codex to search the web and answer the recognized question.",
    )

    speak = subparsers.add_parser("speak", help="Generate MP3 from answer text.")
    speak.add_argument("--text", default=None, help="Text to synthesize.")
    speak.add_argument("--text-file", default=None, help="UTF-8 text file to synthesize.")
    speak.add_argument("--target", choices=["mandarin", "wuu"], default="mandarin", help="Read the text as-is or rewrite it into a Wu oral script first.")
    speak.add_argument("--output", required=True, help="MP3 output path.")
    speak.add_argument("--wu-output", default=None, help="Optional Wu-style MP3 output path generated in addition to --output.")
    speak.add_argument("--wu-text-output", default=None, help="Optional UTF-8 text file for the Wu-style oral script.")
    speak.add_argument("--tts-backend", choices=["edge", "cosyvoice_wu", "command"], default=None)
    speak.add_argument("--wu-backend", choices=["edge", "cosyvoice_wu", "command"], default=None)
    speak.add_argument("--voice", default=DEFAULT_TTS_VOICE, help="edge-tts voice name.")
    speak.add_argument("--wu-voice", default=DEFAULT_WU_TTS_VOICE, help="edge-tts voice for Wu-style output. Use a true wuu-CN voice if available.")
    speak.add_argument("--ref-audio", default=None, help="Reference audio for GPT-SoVITS or external voice cloning.")
    speak.add_argument("--prompt-text", default="", help="Transcript for the reference audio.")
    speak.add_argument("--text-lang", default="zh")
    speak.add_argument("--prompt-lang", default="zh")
    speak.add_argument("--gpt-sovits-url", default=DEFAULT_GPT_SOVITS_URL)
    speak.add_argument("--cosyvoice-wu-url", default=DEFAULT_COSYVOICE_WU_URL)
    speak.add_argument("--tts-command", default=None, help="External command template. Placeholders: {text_file}, {output}, {ref_audio}, {prompt_text}.")
    speak.add_argument("--rate", default="+0%", help="edge-tts speaking rate, for example +0%% or -10%%.")
    speak.add_argument("--pitch", default="+0Hz", help="edge-tts pitch, for example +0Hz.")

    speak_verified = subparsers.add_parser("speak-verified", help="Generate speech, verify it with ASR, and pick the safest candidate.")
    speak_verified.add_argument("--text", default=None, help="Text to synthesize.")
    speak_verified.add_argument("--text-file", default=None, help="UTF-8 text file to synthesize.")
    speak_verified.add_argument("--target", choices=["mandarin", "wuu"], default="wuu")
    speak_verified.add_argument("--output", required=True, help="Final MP3 output path.")
    speak_verified.add_argument("--candidate-dir", default="outputs/tts_candidates", help="Directory for candidate MP3 files and score report.")
    speak_verified.add_argument("--tts-backend", choices=["edge", "cosyvoice_wu", "command"], default="cosyvoice_wu")
    speak_verified.add_argument("--fallback-backend", choices=["edge", "command"], default="edge")
    speak_verified.add_argument("--fallback-text", default=None, help="Clear fallback text used when verified candidates are unsafe.")
    speak_verified.add_argument("--fallback-text-file", default=None, help="UTF-8 fallback text file, usually the Mandarin answer.")
    speak_verified.add_argument("--voice", default=DEFAULT_TTS_VOICE, help="Fallback edge-tts voice name.")
    speak_verified.add_argument("--wu-voice", default=DEFAULT_WU_TTS_VOICE)
    speak_verified.add_argument("--ref-audio", default=None)
    speak_verified.add_argument("--prompt-text", default="")
    speak_verified.add_argument("--reference-experts", default="configs/wu_reference_experts.json")
    speak_verified.add_argument(
        "--reference-gender",
        choices=["auto", "female", "male"],
        default="auto",
    )
    speak_verified.add_argument("--text-langs", default="zh,all_yue", help="Comma-separated GPT-SoVITS text_lang candidates.")
    speak_verified.add_argument("--prompt-lang", default="zh")
    speak_verified.add_argument("--gpt-sovits-url", default=DEFAULT_GPT_SOVITS_URL)
    speak_verified.add_argument("--cosyvoice-wu-url", default=DEFAULT_COSYVOICE_WU_URL)
    speak_verified.add_argument(
        "--secondary-cosyvoice-wu-url",
        default=None,
        help="Optional API-compatible second Wu generator, such as a future CosyVoice3 expert.",
    )
    speak_verified.add_argument("--tts-command", default=None)
    speak_verified.add_argument("--rate", default="+0%")
    speak_verified.add_argument("--pitch", default="+0Hz")
    speak_verified.add_argument("--verify-backend", default=DEFAULT_ASR_BACKEND, choices=ASR_BACKENDS)
    speak_verified.add_argument("--verify-model", default=None)
    speak_verified.add_argument("--min-keyword-recall", type=float, default=0.9)
    speak_verified.add_argument("--min-char-accuracy", type=float, default=0.75)
    speak_verified.add_argument(
        "--unsafe-policy",
        choices=["dual", "fallback", "keep_candidate"],
        default="dual",
        help="What to output when Wu TTS verification is below threshold.",
    )
    speak_verified.add_argument("--tts-review-log", default=DEFAULT_TTS_REVIEW_LOG)
    speak_verified.add_argument("--no-save-tts-review", action="store_true")
    speak_verified.add_argument("--no-prefix-trim", action="store_true")
    speak_verified.add_argument("--json", action="store_true")

    tts_assets = subparsers.add_parser("tts-assets", help="Export single-speaker Wu/Shanghainese assets for voice cloning.")
    tts_assets.add_argument("--output-dir", default="outputs/wu_tts_assets", help="Directory for exported TTS/voice-cloning assets.")
    tts_assets.add_argument("--speaker", default="auto", help="Speaker id, or auto for the speaker with most usable audio.")
    tts_assets.add_argument("--max-items", type=int, default=240)
    tts_assets.add_argument("--max-reference-clips", type=int, default=5)
    tts_assets.add_argument("--min-duration", type=float, default=1.0)
    tts_assets.add_argument("--max-duration", type=float, default=12.0)
    tts_assets.add_argument("--clean-only", action="store_true", help="Prefer cleaner clips with fewer fillers/repetitions for TTS training.")
    tts_assets.add_argument("--max-per-speaker", type=int, default=None, help="Limit exported clips per speaker when --speaker all is used.")
    tts_assets.add_argument("--json", action="store_true")

    batch = subparsers.add_parser("batch", help="Run the agent over a JSONL manifest.")
    batch.add_argument("--manifest", required=True, help="JSONL with audio and optional text/reference fields.")
    batch.add_argument("--output", required=True, help="Output JSONL predictions.")
    batch.add_argument("--backend", default=DEFAULT_ASR_BACKEND, choices=ASR_BACKENDS)
    batch.add_argument("--model", default=None, help="ASR model name or local path.")
    batch.add_argument("--assist-backend", default="none", choices=ASSIST_BACKENDS)
    batch.add_argument("--assist-model", default=None, help="Optional open-source ASR candidate model.")
    batch.add_argument("--glossary", default=None, help="Path to glossary JSON.")
    add_runtime_agent_arguments(batch)
    batch.add_argument(
        "--custom-repair",
        action="append",
        default=[],
        help="Extra post-ASR repair in the form wrong=correct. Can be repeated.",
    )
    batch.add_argument("--local-files-only", action="store_true")
    batch.add_argument("--chunk-seconds", type=float, default=15.0, help="Whisper chunk size for long audio.")
    batch.add_argument("--no-vad", action="store_true", help="Disable pause-based long audio segmentation.")
    batch.add_argument(
        "--max-speech-region-seconds",
        type=float,
        default=8.0,
        help="Maximum speech segment duration when VAD segmentation is enabled.",
    )
    batch.add_argument("--markdown-dir", default=None, help="Optional directory for per-sample reports.")

    evaluate = subparsers.add_parser("evaluate", help="Evaluate a predictions JSONL file.")
    evaluate.add_argument("--predictions", required=True, help="JSONL with reference/text and transcript fields.")
    evaluate.add_argument("--glossary", default=None, help="Path to glossary JSON.")
    evaluate.add_argument("--json", action="store_true", help="Print raw JSON.")

    learning = subparsers.add_parser("learning", help="Inspect or export the active-learning queue.")
    learning.add_argument("--queue", default=DEFAULT_ACTIVE_LEARNING_LOG, help="Active-learning JSONL queue.")
    learning.add_argument("--report", default=None, help="Write a Markdown queue report.")
    learning.add_argument("--export-manifest", default=None, help="Write confirmed items as a training JSONL manifest.")
    learning.add_argument(
        "--include-unconfirmed",
        action="store_true",
        help="Export primary transcripts for unconfirmed items. Use only for inspection, not final training.",
    )
    learning.add_argument("--json", action="store_true", help="Print raw JSON.")

    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "learning":
        return handle_learning_command(args)
    if args.command == "speak":
        return handle_speak_command(args)
    if args.command == "speak-verified":
        return handle_speak_verified_command(args)
    if args.command == "tts-assets":
        return handle_tts_assets_command(args)

    glossary = getattr(args, "glossary", None)
    if glossary is None:
        default_glossary = Path("data/examples/shanghainese_glossary.json")
        glossary = str(default_glossary) if default_glossary.exists() else None

    repair_engine = (
        RepairEngine.from_file(glossary)
        .with_custom_repairs(load_custom_repairs_file(getattr(args, "memory", None)))
        .with_custom_repairs(parse_custom_repairs(getattr(args, "custom_repair", [])))
    )
    if args.command == "evaluate":
        rows = read_jsonl(args.predictions)
        pairs = []
        for row in rows:
            reference = row.get("reference") or row.get("text")
            hypothesis = row.get("transcript") or row.get("hypothesis")
            if reference is not None and hypothesis is not None:
                pairs.append((str(reference), str(hypothesis)))
        summary = evaluate_pairs(
            pairs,
            domain_terms=repair_engine.glossary.get("domain_terms", []),
            dialect_markers=repair_engine.glossary.get("dialect_terms", {}).keys(),
        )
        if args.json:
            print(json.dumps(summary.as_dict(), ensure_ascii=False, indent=2))
        else:
            print_evaluation(summary.as_dict())
        return 0

    backend_name = "mock" if args.command == "demo" else args.backend
    model_name = getattr(args, "model", None)
    if backend_name in {"whisper", "hybrid"} and model_name is None:
        model_name = DEFAULT_ASR_MODEL
    if backend_name in {"funasr", "dolphin", "dolphin_multiagent"} and model_name == DEFAULT_ASR_MODEL:
        model_name = None
    backend = make_backend(
        backend_name,
        model_name=model_name,
        local_files_only=getattr(args, "local_files_only", False),
        chunk_seconds=getattr(args, "chunk_seconds", 15.0),
        vad_enabled=not getattr(args, "no_vad", False),
        max_speech_region_seconds=getattr(args, "max_speech_region_seconds", None),
        assist_backend=getattr(args, "assist_backend", "none"),
        assist_model=getattr(args, "assist_model", None),
    )
    agent = ShanghaiDialectAgent(asr_backend=backend, repair_engine=repair_engine)

    if args.command == "batch":
        rows = run_batch(
            agent,
            args.manifest,
            args.markdown_dir,
            active_learning_log=getattr(args, "active_learning_log", None),
            save_active_learning=not getattr(args, "no_save_active_learning", False),
        )
        write_jsonl(args.output, rows)
        print(f"Wrote {len(rows)} predictions to {args.output}")
        return 0

    result = agent.run(audio_path=getattr(args, "audio", None))
    active_learning_saved = 0
    if args.command in {"transcribe", "translate"} and not getattr(args, "no_save_active_learning", False):
        active_learning_saved = append_active_learning_items(
            getattr(args, "active_learning_log", None),
            result.active_learning_items,
        )

    if getattr(args, "markdown", None):
        markdown_path = Path(args.markdown)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(render_markdown_report(result), encoding="utf-8")

    if args.command == "translate":
        product = build_translation_product(result)
        codex_task_payload = None
        if getattr(args, "codex_task_output", None):
            codex_task_path = Path(args.codex_task_output)
            codex_task_path.parent.mkdir(parents=True, exist_ok=True)
            codex_task_path.write_text(
                render_codex_answer_task(product, result, audio_path=getattr(args, "audio", None)),
                encoding="utf-8",
            )
            codex_task_payload = render_codex_task_metadata(product, codex_task_path)
        tts_payload = None
        if getattr(args, "tts_output", None):
            tts_backend = args.tts_backend or (DEFAULT_WU_TTS_BACKEND if args.tts_target == "wuu" else DEFAULT_TTS_BACKEND)
            tts_text = resolve_tts_text(args.tts_target, product.mandarin, backend=tts_backend)
            tts_path = synthesize_mp3(
                TTSRequest(
                    text=tts_text,
                    output_path=args.tts_output,
                    voice=args.tts_voice,
                    rate=args.tts_rate,
                    pitch=args.tts_pitch,
                    backend=tts_backend,
                    ref_audio_path=args.ref_audio,
                    prompt_text=args.prompt_text,
                    text_lang=args.text_lang,
                    prompt_lang=args.prompt_lang,
                    cosyvoice_wu_url=args.cosyvoice_wu_url,
                    gpt_sovits_url=args.gpt_sovits_url,
                    command_template=args.tts_command,
                )
            )
            tts_payload = {
                "target": args.tts_target,
                "voice": args.tts_voice,
                "backend": tts_backend,
                "text": tts_text,
                "output": str(tts_path),
            }
        if args.json:
            payload = product.as_dict()
            payload["repairs"] = result.repairs
            payload["suspicions"] = [item.__dict__ for item in result.suspicions]
            payload["alternatives"] = result.alternatives
            payload["agent_trace"] = result.agent_trace
            payload["active_learning_items"] = result.active_learning_items
            payload["active_learning_saved"] = active_learning_saved
            if codex_task_payload:
                payload["codex_task"] = codex_task_payload
            if tts_payload:
                payload["tts"] = tts_payload
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print_translation(product.as_dict())
            if codex_task_payload:
                print()
                print(f"Codex 联网问答任务: {codex_task_payload['codex_task_output']}")
            if tts_payload:
                print()
                print(f"语音输出: {tts_payload['output']}")
                print(f"语音内容: {tts_payload['text']}")
        return 0

    if args.json:
        payload = result.as_dict()
        payload["active_learning_saved"] = active_learning_saved
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print_human_report(result.as_dict())
    return 0


def print_human_report(result: dict) -> None:
    dialect = result["dialect"]
    print("Shanghai Dialect ASR Repair Agent")
    print("=" * 32)
    print(f"Dialect: {dialect['label']}  score={dialect['score']}  markers={', '.join(dialect['markers'])}")
    print()
    print("Transcript")
    print(result["transcript"])
    print()
    print("Mandarin translation")
    print(result["mandarin_translation"])
    print()
    print("Repairs")
    if result["repairs"]:
        for item in result["repairs"]:
            print(f"- segment {item['segment_index']}: {item['original']} -> {item['repaired']}")
    else:
        print("- none")


def print_translation(product: dict) -> None:
    print("普通话结果")
    print("=" * 16)
    print(product["mandarin"])
    print()
    print("识别原文")
    print(product["dialect_transcript"])
    print()
    print(f"状态: {product.get('status_label', product['status'])}")
    if product.get("warning"):
        print(f"提示: {product['warning']}")
    if product.get("draft_mandarin"):
        print()
        print("高风险草稿")
        print(product["draft_mandarin"])
    print()
    print(f"修复次数: {product.get('repair_count', 0)}")
    print(f"可疑片段: {product.get('suspicion_count', 0)}")
    print(f"质量评分: {product.get('quality_score', 'n/a')}")
    if product.get("consensus_score") is not None:
        print(f"候选共识: {product['consensus_score']}")
    if product.get("action_suggestion"):
        print(f"建议: {product['action_suggestion']}")


def run_batch(
    agent: ShanghaiDialectAgent,
    manifest: str,
    markdown_dir: str | None = None,
    active_learning_log: str | None = None,
    save_active_learning: bool = True,
) -> list[dict]:
    rows = read_jsonl(manifest)
    outputs: list[dict] = []
    report_dir = Path(markdown_dir) if markdown_dir else None
    if report_dir:
        report_dir.mkdir(parents=True, exist_ok=True)

    for index, row in enumerate(rows):
        audio = row.get("audio")
        result = agent.run(audio_path=str(audio) if audio else None)
        product = build_translation_product(result)
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
            "status": product.status,
            "status_label": product.status_label,
            "quality_score": product.quality_score,
            "consensus_score": product.consensus_score,
            "action_suggestion": product.action_suggestion,
            "repairs": result.repairs,
            "suspicions": [item.__dict__ for item in result.suspicions],
            "alternatives": result.alternatives,
            "agent_trace": result.agent_trace,
            "active_learning_items": result.active_learning_items,
            "active_learning_saved": append_active_learning_items(active_learning_log, result.active_learning_items)
            if save_active_learning
            else 0,
        }
        outputs.append(output)

        if report_dir:
            report_path = report_dir / f"sample_{index:04d}.md"
            report_path.write_text(render_markdown_report(result), encoding="utf-8")
    return outputs


def print_evaluation(summary: dict) -> None:
    print("Evaluation")
    print("=" * 16)
    print(f"Samples: {summary['sample_count']}")
    print(f"CER: {summary['cer']}")
    print(f"Term recall: {summary['term_recall']}")
    print(f"Dialect marker recall: {summary['dialect_marker_recall']}")
    print(f"Exact match rate: {summary['exact_match_rate']}")


def handle_speak_command(args: argparse.Namespace) -> int:
    text = args.text or ""
    if args.text_file:
        text = Path(args.text_file).read_text(encoding="utf-8")
    if not text.strip():
        raise SystemExit("speak requires --text or --text-file with non-empty content.")

    primary_backend = (
        args.wu_backend or args.tts_backend or DEFAULT_WU_TTS_BACKEND
        if args.target == "wuu"
        else args.tts_backend or DEFAULT_TTS_BACKEND
    )
    tts_text = resolve_tts_text(args.target, text, backend=primary_backend)
    primary_voice = args.wu_voice if args.target == "wuu" else args.voice
    output_path = synthesize_mp3(
        TTSRequest(
            text=tts_text,
            output_path=args.output,
            voice=primary_voice,
            rate=args.rate,
            pitch=args.pitch,
            backend=primary_backend,
            ref_audio_path=args.ref_audio,
            prompt_text=args.prompt_text,
            text_lang=args.text_lang,
            prompt_lang=args.prompt_lang,
            cosyvoice_wu_url=args.cosyvoice_wu_url,
            gpt_sovits_url=args.gpt_sovits_url,
            command_template=args.tts_command,
        )
    )
    print(f"语音输出: {output_path}")
    if args.target == "wuu":
        if args.wu_text_output:
            wu_text_path = Path(args.wu_text_output)
            wu_text_path.parent.mkdir(parents=True, exist_ok=True)
            wu_text_path.write_text(tts_text, encoding="utf-8")
        print("吴语口语稿")
        print(tts_text)
        notice = wu_voice_notice(primary_voice, primary_backend)
        if notice:
            print(f"提示: {notice}")
    if args.wu_output:
        wu_backend = args.wu_backend or args.tts_backend or DEFAULT_WU_TTS_BACKEND
        wu_text = resolve_tts_text("wuu", text, backend=wu_backend)
        if args.wu_text_output:
            wu_text_path = Path(args.wu_text_output)
            wu_text_path.parent.mkdir(parents=True, exist_ok=True)
            wu_text_path.write_text(wu_text, encoding="utf-8")
        wu_output_path = synthesize_mp3(
            TTSRequest(
                text=wu_text,
                output_path=args.wu_output,
                voice=args.wu_voice,
                rate=args.rate,
                pitch=args.pitch,
                backend=wu_backend,
                ref_audio_path=args.ref_audio,
                prompt_text=args.prompt_text,
                text_lang=args.text_lang,
                prompt_lang=args.prompt_lang,
                cosyvoice_wu_url=args.cosyvoice_wu_url,
                gpt_sovits_url=args.gpt_sovits_url,
                command_template=args.tts_command,
            )
        )
        print(f"吴语语音输出: {wu_output_path}")
        print("吴语口语稿")
        print(wu_text)
        notice = wu_voice_notice(args.wu_voice, args.wu_backend or args.tts_backend or DEFAULT_WU_TTS_BACKEND)
        if notice:
            print(f"提示: {notice}")
    return 0


def handle_speak_verified_command(args: argparse.Namespace) -> int:
    text = args.text or ""
    if args.text_file:
        text = Path(args.text_file).read_text(encoding="utf-8")
    if not text.strip():
        raise SystemExit("speak-verified requires --text or --text-file with non-empty content.")
    fallback_text = args.fallback_text or ""
    if args.fallback_text_file:
        fallback_text = Path(args.fallback_text_file).read_text(encoding="utf-8")
    if not fallback_text.strip():
        fallback_text = text

    candidate_dir = Path(args.candidate_dir)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    verify_backend = make_backend(
        args.verify_backend,
        model_name=args.verify_model,
        chunk_seconds=15.0,
        vad_enabled=True,
        max_speech_region_seconds=8.0,
    )
    verify_agent = ShanghaiDialectAgent(asr_backend=verify_backend, repair_engine=RepairEngine())

    tts_text = resolve_tts_text(args.target, text, backend=args.tts_backend)
    sentence_texts = split_tts_sentences(
        tts_text,
        normalize_numbers=args.tts_backend != "cosyvoice_wu",
    )
    if not sentence_texts:
        raise SystemExit("TTS text became empty after sentence splitting.")

    sentence_reports: list[dict] = []
    selected_sentence_paths: list[Path] = []
    all_candidates: list[dict] = []
    cosy_speeds = [0.94, 0.88, 1.0, 0.82, 0.9]
    candidate_seeds = [1986, 2026, 3407, 42, 8675309]
    prefix_trim_enabled = not args.no_prefix_trim and allows_prefix_trim(tts_text)
    reference_experts: list[WuReferenceExpert] = []
    if args.tts_backend == "cosyvoice_wu":
        if args.ref_audio:
            reference_experts = [
                WuReferenceExpert(
                    expert_id="manual",
                    audio=str(Path(args.ref_audio).resolve()),
                    prompt_text=args.prompt_text,
                    gender=args.reference_gender,
                    domains=("general",),
                    quality=1.0,
                    use_server_default=False,
                )
            ]
        else:
            reference_experts = select_wu_reference_experts(
                tts_text,
                load_wu_reference_experts(args.reference_experts),
                gender=args.reference_gender,
            )
    generation_endpoints = [("primary", args.cosyvoice_wu_url)]
    if args.secondary_cosyvoice_wu_url:
        generation_endpoints.append(("secondary", args.secondary_cosyvoice_wu_url))
    baseline_reference = reference_experts[0] if reference_experts else None
    baseline_schedule = [
        (
            "primary",
            args.cosyvoice_wu_url,
            baseline_reference,
            cosy_speeds[index],
            candidate_seeds[index],
        )
        for index in range(len(candidate_seeds))
    ]
    generation_schedule = baseline_schedule[:2]
    generation_schedule.extend(
        ("primary", args.cosyvoice_wu_url, reference, cosy_speeds[0], candidate_seeds[0])
        for reference in reference_experts[1:]
    )
    if args.secondary_cosyvoice_wu_url:
        generation_schedule.append(
            (
                "secondary",
                args.secondary_cosyvoice_wu_url,
                baseline_reference,
                cosy_speeds[0],
                candidate_seeds[0],
            )
        )
    generation_schedule.extend(baseline_schedule[2:])
    generation_schedule = generation_schedule[:8]
    for sentence_index, sentence_text in enumerate(sentence_texts, start=1):
        sentence_dir = candidate_dir / f"sentence_{sentence_index:02d}"
        sentence_dir.mkdir(parents=True, exist_ok=True)
        sentence_candidates: list[dict] = []
        attempts = len(generation_schedule) if args.tts_backend == "cosyvoice_wu" else 1
        for attempt in range(1, attempts + 1):
            text_lang = "wu" if args.tts_backend == "cosyvoice_wu" else "zh"
            candidate_path = sentence_dir / f"candidate_{attempt:02d}.wav"
            endpoint_id, endpoint_url, reference_expert, candidate_speed, candidate_seed = (
                generation_schedule[attempt - 1]
            )
            synthesize_mp3(
                TTSRequest(
                    text=sentence_text,
                    output_path=candidate_path,
                    voice=args.wu_voice if args.target == "wuu" else args.voice,
                    rate=args.rate,
                    pitch=args.pitch,
                    backend=args.tts_backend,
                    ref_audio_path=(
                        None
                        if reference_expert and reference_expert.use_server_default
                        else reference_expert.audio
                        if reference_expert
                        else args.ref_audio
                    ),
                    prompt_text=(
                        reference_expert.prompt_text if reference_expert else args.prompt_text
                    ),
                    text_lang=text_lang,
                    prompt_lang=args.prompt_lang,
                    cosyvoice_wu_url=endpoint_url,
                    gpt_sovits_url=args.gpt_sovits_url,
                    command_template=args.tts_command,
                    speed=candidate_speed if args.tts_backend == "cosyvoice_wu" else 1.0,
                    seed=candidate_seed,
                )
            )
            verification = verify_agent.run(audio_path=str(candidate_path))
            quality = score_spoken_answer(
                sentence_text,
                verification.transcript,
                mandarin_translation=verification.mandarin_translation,
                dialect_score=verification.dialect.score,
                suspicion_count=len(verification.suspicions),
            )
            candidate = {
                "path": str(candidate_path),
                "text_lang": text_lang,
                "attempt": attempt,
                "speed": candidate_speed if args.tts_backend == "cosyvoice_wu" else 1.0,
                "seed": candidate_seed,
                "generator": endpoint_id,
                "generator_url": endpoint_url,
                "reference_expert": (
                    reference_expert.expert_id if reference_expert else "server_default"
                ),
                "reference_gender": (
                    reference_expert.gender if reference_expert else "unknown"
                ),
                "quality": quality.as_dict(),
                "transcript": verification.transcript,
                "mandarin_translation": verification.mandarin_translation,
            }
            sentence_candidates.append(candidate)
            all_candidates.append(candidate)
            accepted_candidate = candidate
            detected_prefix_chars = leading_hallucination_chars(
                sentence_text,
                verification.transcript,
            )
            if args.tts_backend == "cosyvoice_wu" and prefix_trim_enabled:
                trimmed_path = sentence_dir / f"candidate_{attempt:02d}_trimmed.wav"
                trim_info = trim_wav_leading_hallucination(
                    candidate_path,
                    trimmed_path,
                    sentence_text,
                    verification.transcript,
                )
                if trim_info:
                    trimmed_verification = verify_agent.run(audio_path=str(trimmed_path))
                    trimmed_quality = score_spoken_answer(
                        sentence_text,
                        trimmed_verification.transcript,
                        mandarin_translation=trimmed_verification.mandarin_translation,
                        dialect_score=trimmed_verification.dialect.score,
                        suspicion_count=len(trimmed_verification.suspicions),
                    )
                    trimmed_candidate = {
                        **candidate,
                        "path": str(trimmed_path),
                        "variant": "prefix_trimmed",
                        "trim": trim_info,
                        "quality": trimmed_quality.as_dict(),
                        "transcript": trimmed_verification.transcript,
                        "mandarin_translation": trimmed_verification.mandarin_translation,
                    }
                    if (
                        trimmed_quality.passes_critical_gate
                        and trimmed_quality.keyword_recall >= quality.keyword_recall
                        and trimmed_quality.char_accuracy > quality.char_accuracy
                    ):
                        sentence_candidates.append(trimmed_candidate)
                        all_candidates.append(trimmed_candidate)
                        accepted_candidate = trimmed_candidate
            if (
                accepted_candidate["quality"]["passes_critical_gate"]
                and accepted_candidate["quality"]["keyword_recall"] >= args.min_keyword_recall
                and accepted_candidate["quality"]["char_accuracy"] >= args.min_char_accuracy
            ):
                high_quality = (
                    accepted_candidate["quality"]["char_accuracy"] >= 0.85
                    and detected_prefix_chars == 0
                )
                explored_reference = attempt >= min(3, attempts)
                if high_quality or len(reference_experts) <= 1 or explored_reference:
                    break

        best = max(
            sentence_candidates,
            key=lambda item: (
                item["quality"]["passes_critical_gate"],
                item["quality"]["critical_entity_recall"],
                item["quality"]["keyword_recall"],
                item["quality"]["char_accuracy"],
                item["quality"]["dialect_score"],
                -item["quality"]["suspicion_count"],
            ),
        )
        sentence_unsafe = (
            not best["quality"]["passes_critical_gate"]
            or best["quality"]["keyword_recall"] < args.min_keyword_recall
            or best["quality"]["char_accuracy"] < args.min_char_accuracy
        )
        selected_sentence_paths.append(Path(best["path"]))
        sentence_reports.append(
            {
                "index": sentence_index,
                "text": sentence_text,
                "selected": best,
                "unsafe": sentence_unsafe,
                "attempts": len(sentence_candidates),
                "candidates": sentence_candidates,
            }
        )

    missing_terms = _dedupe_strings(
        term
        for sentence in sentence_reports
        for term in sentence["selected"]["quality"]["missing_terms"]
    )
    missing_critical_terms = _dedupe_strings(
        term
        for sentence in sentence_reports
        for term in sentence["selected"]["quality"]["missing_critical_terms"]
    )
    critical_gate_failed = bool(missing_critical_terms)
    unsafe_candidate = any(sentence["unsafe"] for sentence in sentence_reports)
    # Critical entities are a hard gate: never publish a reply with a missing
    # phone number, certificate, police-station, or household-registration term.
    used_fallback = critical_gate_failed or (
        unsafe_candidate and args.unsafe_policy == "fallback"
    )
    fallback_output_path: Path | None = None
    if used_fallback:
        synthesize_mp3(
            TTSRequest(
                text=fallback_text,
                output_path=output_path,
                voice=args.voice,
                rate=args.rate,
                pitch=args.pitch,
                backend=args.fallback_backend,
                command_template=args.tts_command,
            )
        )
    else:
        # Verified clauses need a clear boundary. A longer pause prevents Wu ASR
        # from merging a hotline number with the first syllable of the next clause.
        join_wav_files(selected_sentence_paths, output_path, pause_seconds=0.55)
        if unsafe_candidate and args.unsafe_policy == "dual":
            fallback_output_path = output_path.with_name(f"{output_path.stem}_safe_mandarin{output_path.suffix}")
            synthesize_mp3(
                TTSRequest(
                    text=fallback_text,
                    output_path=fallback_output_path,
                    voice=args.voice,
                    rate=args.rate,
                    pitch=args.pitch,
                    backend=args.fallback_backend,
                    command_template=args.tts_command,
                )
            )

    selected = {
        "path": str(output_path),
        "text_lang": "wu" if args.tts_backend == "cosyvoice_wu" else "zh",
        "transcript": "".join(
            sentence["selected"]["transcript"] for sentence in sentence_reports
        ),
        "mandarin_translation": "".join(
            sentence["selected"]["mandarin_translation"] for sentence in sentence_reports
        ),
        "quality": {
            "keyword_recall": min(
                sentence["selected"]["quality"]["keyword_recall"]
                for sentence in sentence_reports
            ),
            "char_accuracy": min(
                sentence["selected"]["quality"]["char_accuracy"]
                for sentence in sentence_reports
            ),
            "dialect_score": round(
                sum(sentence["selected"]["quality"]["dialect_score"] for sentence in sentence_reports)
                / len(sentence_reports),
                4,
            ),
            "suspicion_count": sum(
                sentence["selected"]["quality"]["suspicion_count"]
                for sentence in sentence_reports
            ),
            "missing_terms": missing_terms,
            "matched_terms": _dedupe_strings(
                term
                for sentence in sentence_reports
                for term in sentence["selected"]["quality"]["matched_terms"]
            ),
            "missing_critical_terms": missing_critical_terms,
            "passes_critical_gate": not critical_gate_failed,
        },
    }
    report = {
        "output": str(output_path),
        "target": args.target,
        "selected": selected,
        "unsafe_candidate": unsafe_candidate,
        "critical_gate_failed": critical_gate_failed,
        "unsafe_policy": args.unsafe_policy,
        "used_fallback": used_fallback,
        "fallback_output": str(fallback_output_path) if fallback_output_path else (str(output_path) if used_fallback else None),
        "fallback_text_source": args.fallback_text_file or ("--fallback-text" if args.fallback_text else "primary_text"),
        "min_keyword_recall": args.min_keyword_recall,
        "min_char_accuracy": args.min_char_accuracy,
        "sentence_count": len(sentence_reports),
        "reference_experts": [expert.expert_id for expert in reference_experts],
        "generation_endpoints": [item[0] for item in generation_endpoints],
        "prefix_trim_enabled": prefix_trim_enabled,
        "sentences": sentence_reports,
        "candidates": all_candidates,
    }
    review_saved = 0
    if unsafe_candidate and not args.no_save_tts_review:
        review_saved = append_tts_review_item(args.tts_review_log, report, source_text=tts_text, fallback_text=fallback_text)
        report["tts_review_saved"] = review_saved
        report["tts_review_log"] = args.tts_review_log
    report_path = candidate_dir / "verification_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"Verified speech output: {output_path}")
        print(f"Report: {report_path}")
        print(f"Selected sentence candidates: {len(selected_sentence_paths)}")
        print(f"Minimum sentence keyword recall: {selected['quality']['keyword_recall']}")
        if critical_gate_failed:
            print(f"Critical entity gate failed: {', '.join(missing_critical_terms)}")
        if unsafe_candidate and args.unsafe_policy == "dual":
            print(f"Unsafe Wu candidate kept for demo; safe Mandarin fallback: {fallback_output_path}")
            print(f"TTS review saved: {review_saved} -> {args.tts_review_log}")
        elif used_fallback:
            print("Fallback used: keyword recall below threshold, generated a clearer Mandarin/edge-tts output.")
            print(f"TTS review saved: {review_saved} -> {args.tts_review_log}")
        elif unsafe_candidate:
            print("Unsafe Wu candidate kept by policy; no Mandarin fallback was generated.")
            print(f"TTS review saved: {review_saved} -> {args.tts_review_log}")
    return 0


def _dedupe_strings(items) -> list[str]:
    return list(dict.fromkeys(item for item in items if item))


def append_tts_review_item(log_path: str | Path, report: dict, *, source_text: str, fallback_text: str) -> int:
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    selected = report["selected"]
    quality = selected["quality"]
    item = {
        "type": "tts_quality_review",
        "output": report["output"],
        "candidate_path": selected["path"],
        "text_lang": selected["text_lang"],
        "source_text": source_text,
        "fallback_text": fallback_text,
        "keyword_recall": quality["keyword_recall"],
        "dialect_score": quality["dialect_score"],
        "missing_terms": quality["missing_terms"],
        "missing_critical_terms": quality.get("missing_critical_terms", []),
        "passes_critical_gate": quality.get("passes_critical_gate", True),
        "matched_terms": quality["matched_terms"],
        "transcript": selected["transcript"],
        "suggested_action": "人工听评候选音频，确认缺失关键词后补入吴语 TTS 评估集或下一轮训练清单。",
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    return 1


def handle_tts_assets_command(args: argparse.Namespace) -> int:
    export = export_voice_clone_assets(
        output_dir=args.output_dir,
        speaker_id=args.speaker,
        max_items=args.max_items,
        max_reference_clips=args.max_reference_clips,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        clean_only=args.clean_only,
        max_per_speaker=args.max_per_speaker,
    )
    payload = export.as_dict()
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print("Wu/Shanghainese Voice Cloning Assets")
        print("=" * 38)
        print(f"Output: {payload['output_dir']}")
        print(f"Speaker: {payload['speaker_id']}")
        print(f"Items: {payload['item_count']}")
        print(f"Duration minutes: {payload['duration_minutes']}")
        print(f"Reference audio: {payload['reference_audio']}")
        print(f"Reference text: {payload['reference_text']}")
        print(f"GPT-SoVITS list: {payload['gpt_sovits_list']}")
        print(f"CosyVoice manifest: {payload['cosyvoice_manifest']}")
    return 0


def handle_learning_command(args: argparse.Namespace) -> int:
    records = read_active_learning_queue(args.queue)
    summary = summarize_active_learning_items(records)
    exported = None
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(render_active_learning_report(records), encoding="utf-8")
    if args.export_manifest:
        exported = export_active_learning_manifest(
            args.queue,
            args.export_manifest,
            include_unconfirmed=args.include_unconfirmed,
        )

    payload = {
        "queue": args.queue,
        "summary": summary.as_dict(),
        "report": args.report,
        "export_manifest": args.export_manifest,
        "exported_rows": exported,
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print_learning_summary(payload)
    return 0


def print_learning_summary(payload: dict) -> None:
    summary = payload["summary"]
    print("Active Learning Queue")
    print("=" * 22)
    print(f"Queue: {payload['queue']}")
    print(f"Total: {summary['total']}")
    print(f"Pending review: {summary['pending']}")
    print(f"Confirmed: {summary['confirmed']}")
    print(f"Export-ready: {summary['exported_ready']}")
    print(f"Unique audio files: {summary['unique_audio']}")
    if summary["reason_counts"]:
        print("Reasons:")
        for reason, count in summary["reason_counts"].items():
            print(f"- {reason}: {count}")
    if payload.get("report"):
        print(f"Report: {payload['report']}")
    if payload.get("export_manifest"):
        print(f"Exported rows: {payload.get('exported_rows', 0)} -> {payload['export_manifest']}")


if __name__ == "__main__":
    raise SystemExit(main())
