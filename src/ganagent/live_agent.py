from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from ganagent.agent import ShanghaiDialectAgent
from ganagent.asr_backends import make_backend
from ganagent.audio_capture import (
    RecordingConfig,
    calibrate_noise_floor,
    list_input_devices,
    play_audio,
    record_utterance,
)
from ganagent.cli import (
    ASSIST_BACKENDS,
    ASR_BACKENDS,
    DEFAULT_ASR_BACKEND,
    DEFAULT_ASR_MODEL,
    DEFAULT_COSYVOICE_WU_URL,
    DEFAULT_REPAIR_MEMORY,
    DEFAULT_TTS_BACKEND,
    DEFAULT_TTS_VOICE,
    DEFAULT_WU_TTS_BACKEND,
    DEFAULT_WU_TTS_VOICE,
)
from ganagent.codex_task import render_codex_answer_task
from ganagent.dialogue_manager import build_dialogue_reply
from ganagent.learning import append_active_learning_items
from ganagent.models import AgentResult, DialectSignal, Segment
from ganagent.product import build_translation_product
from ganagent.product import TranslationProduct
from ganagent.repair import RepairEngine, load_custom_repairs_file, parse_custom_repairs
from ganagent.tts import TTSRequest, resolve_tts_text, synthesize_mp3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Realtime turn-based Shanghai dialect dialogue agent."
    )
    parser.add_argument("--backend", default=DEFAULT_ASR_BACKEND, choices=ASR_BACKENDS)
    parser.add_argument("--model", default=None)
    parser.add_argument("--assist-backend", default="none", choices=ASSIST_BACKENDS)
    parser.add_argument("--assist-model", default=None)
    parser.add_argument("--glossary", default=None)
    parser.add_argument("--memory", default=DEFAULT_REPAIR_MEMORY)
    parser.add_argument("--custom-repair", action="append", default=[])
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--chunk-seconds", type=float, default=15.0)
    parser.add_argument("--max-speech-region-seconds", type=float, default=8.0)
    parser.add_argument("--no-vad", action="store_true")
    parser.add_argument("--turns", type=int, default=0, help="0 means keep listening until Ctrl+C.")
    parser.add_argument("--output-dir", default="outputs/live_agent")
    parser.add_argument("--active-learning-log", default="data/active_learning_queue.jsonl")
    parser.add_argument("--no-save-active-learning", action="store_true")
    parser.add_argument("--codex-task-dir", default="outputs/live_agent")
    parser.add_argument("--reply-target", choices=["mandarin", "wuu"], default="wuu")
    parser.add_argument("--tts-backend", choices=["edge", "cosyvoice_wu", "command"], default=None)
    parser.add_argument("--fallback-to-edge", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tts-voice", default=DEFAULT_TTS_VOICE)
    parser.add_argument("--wu-voice", default=DEFAULT_WU_TTS_VOICE)
    parser.add_argument("--tts-rate", default="+0%")
    parser.add_argument("--tts-pitch", default="+0Hz")
    parser.add_argument("--cosyvoice-wu-url", default=DEFAULT_COSYVOICE_WU_URL)
    parser.add_argument("--tts-command", default=None)
    parser.add_argument("--ref-audio", default=None)
    parser.add_argument("--prompt-text", default="")
    parser.add_argument("--prompt-lang", default="zh")
    parser.add_argument("--list-devices", action="store_true", help="List microphone input devices and exit.")
    parser.add_argument("--input-device", default=None, help="sounddevice input device index or name.")
    parser.add_argument("--calibrate-noise", action="store_true", help="Measure ambient noise before listening.")
    parser.add_argument(
        "--text-turn",
        action="append",
        default=[],
        help="Simulate one dialogue turn from text. Can be repeated for mic-free demos.",
    )
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--max-record-seconds", type=float, default=18.0)
    parser.add_argument("--silence-seconds", type=float, default=0.85)
    parser.add_argument("--start-threshold", type=float, default=0.012)
    parser.add_argument("--stop-threshold", type=float, default=0.007)
    parser.add_argument("--no-tts", action="store_true", help="Skip speech synthesis and only print/save the reply.")
    parser.add_argument("--no-playback", action="store_true")
    parser.add_argument("--json-log", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_devices:
        _print_input_devices()
        return 0
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    agent = None if args.text_turn else _build_agent(args)
    input_device = _parse_input_device(args.input_device)
    start_threshold = args.start_threshold
    stop_threshold = args.stop_threshold
    if args.calibrate_noise and not args.text_turn:
        calibration = calibrate_noise_floor(
            sample_rate=args.sample_rate,
            device=input_device,
        )
        start_threshold = calibration["start_threshold"]
        stop_threshold = calibration["stop_threshold"]
        print(
            "麦克风环境噪声校准："
            f"rms={calibration['noise_rms']}, "
            f"start={start_threshold}, stop={stop_threshold}"
        )
    record_config = RecordingConfig(
        sample_rate=args.sample_rate,
        device=input_device,
        max_seconds=args.max_record_seconds,
        silence_seconds=args.silence_seconds,
        start_threshold=start_threshold,
        stop_threshold=stop_threshold,
    )

    print("上海话实时对话 Agent 已启动。")
    print("直接对着麦克风说话；说完停顿一下即可触发识别。按 Ctrl+C 退出。")
    if args.text_turn:
        print("当前为文本模拟模式：不调用麦克风和 ASR，用于课堂无设备演示。")
    print(f"输出目录：{output_dir.resolve()}")
    turn = 0
    total_turns = len(args.text_turn) if args.text_turn and args.turns <= 0 else args.turns
    try:
        while total_turns <= 0 or turn < total_turns:
            turn += 1
            print(f"\n[第 {turn} 轮] 正在监听...")
            if args.text_turn:
                simulated_text = args.text_turn[(turn - 1) % len(args.text_turn)]
                recording = None
                result, product = build_simulated_turn(simulated_text)
                print(f"文本模拟输入：{simulated_text}")
            else:
                audio_path = output_dir / f"turn_{turn:03d}_user.wav"
                recording = record_utterance(audio_path, config=record_config)
                if not recording.speech_started:
                    print("没有检测到清晰语音，继续监听。")
                    continue
                print(
                    f"检测到语音 {recording.duration_seconds:.2f}s，峰值 {recording.peak_rms:.4f}，正在识别..."
                )
                if agent is None:
                    raise RuntimeError("ASR agent was not initialized.")
                result = agent.run(audio_path=str(recording.path))
                product = build_translation_product(result)
            active_learning_saved = 0
            if not args.no_save_active_learning:
                active_learning_saved = append_active_learning_items(
                    args.active_learning_log,
                    result.active_learning_items,
                )
            reply = build_dialogue_reply(product, result)
            codex_task_path = None
            if reply.needs_codex_search:
                codex_task_dir = Path(args.codex_task_dir)
                codex_task_dir.mkdir(parents=True, exist_ok=True)
                codex_task_path = codex_task_dir / f"turn_{turn:03d}_codex_task.md"
                codex_task_path.write_text(
                    render_codex_answer_task(
                        product,
                        result,
                        audio_path=recording.path if recording else None,
                    ),
                    encoding="utf-8",
                )

            print(f"识别原文：{product.dialect_transcript}")
            print(f"普通话理解：{product.mandarin}")
            print(f"回答来源：{reply.source}")
            print(f"Agent 回复：{reply.text}")
            if codex_task_path:
                print(f"已生成 Codex 联网任务：{codex_task_path}")

            reply_audio = None if args.no_tts else _speak_reply(args, reply.text, output_dir, turn)
            if reply_audio and not args.no_playback:
                print(f"正在播放：{reply_audio}")
                play_audio(reply_audio)

            turn_report = {
                "turn": turn,
                "audio": str(recording.path) if recording else None,
                "recording": _recording_payload(recording),
                "product": product.as_dict(),
                "reply": reply.__dict__,
                "reply_audio": str(reply_audio) if reply_audio else None,
                "codex_task": str(codex_task_path) if codex_task_path else None,
                "active_learning_saved": active_learning_saved,
                "timestamp": time.time(),
            }
            _write_turn_report(output_dir, turn, turn_report)
            _append_json_log(
                args.json_log,
                turn_report,
            )
    except KeyboardInterrupt:
        print("\n已退出实时对话。")
    return 0


def _build_agent(args: argparse.Namespace) -> ShanghaiDialectAgent:
    glossary = args.glossary
    if glossary is None:
        default_glossary = Path("data/examples/shanghainese_glossary.json")
        glossary = str(default_glossary) if default_glossary.exists() else None
    repair_engine = (
        RepairEngine.from_file(glossary)
        .with_custom_repairs(load_custom_repairs_file(args.memory))
        .with_custom_repairs(parse_custom_repairs(args.custom_repair))
    )
    model_name = args.model
    if args.backend in {"whisper", "hybrid"} and model_name is None:
        model_name = DEFAULT_ASR_MODEL
    if args.backend in {"funasr", "dolphin", "dolphin_multiagent"} and model_name == DEFAULT_ASR_MODEL:
        model_name = None
    backend = make_backend(
        args.backend,
        model_name=model_name,
        local_files_only=args.local_files_only,
        chunk_seconds=args.chunk_seconds,
        vad_enabled=not args.no_vad,
        max_speech_region_seconds=args.max_speech_region_seconds,
        assist_backend=args.assist_backend,
        assist_model=args.assist_model,
    )
    return ShanghaiDialectAgent(asr_backend=backend, repair_engine=repair_engine)


def build_simulated_turn(text: str) -> tuple[AgentResult, TranslationProduct]:
    result = AgentResult(
        audio_path=None,
        dialect=DialectSignal(label="simulated_text", score=0.0, markers=[]),
        segments=[Segment(0.0, 1.0, text, confidence=1.0, backend="text_turn")],
        suspicions=[],
        transcript=text,
        mandarin_translation=text,
        agent_trace=[
            {
                "agent": "文本模拟输入",
                "role": "课堂演示",
                "status": "ok",
                "summary": "跳过麦克风和 ASR，用指定文本模拟一轮实时对话。",
            }
        ],
    )
    return result, build_translation_product(result)


def _speak_reply(args: argparse.Namespace, text: str, output_dir: Path, turn: int) -> Path | None:
    if not text.strip():
        return None
    backend = args.tts_backend or (
        DEFAULT_WU_TTS_BACKEND if args.reply_target == "wuu" else DEFAULT_TTS_BACKEND
    )
    target = args.reply_target
    tts_text = resolve_tts_text(target, text, backend=backend)
    output = output_dir / f"turn_{turn:03d}_reply_{target}.mp3"
    try:
        return synthesize_mp3(
            TTSRequest(
                text=tts_text,
                output_path=output,
                voice=args.wu_voice if target == "wuu" else args.tts_voice,
                rate=args.tts_rate,
                pitch=args.tts_pitch,
                backend=backend,
                ref_audio_path=args.ref_audio,
                prompt_text=args.prompt_text,
                text_lang="wu" if backend == "cosyvoice_wu" else "zh",
                prompt_lang=args.prompt_lang,
                cosyvoice_wu_url=args.cosyvoice_wu_url,
                command_template=args.tts_command,
            )
        )
    except Exception as exc:
        if not args.fallback_to_edge or backend == "edge":
            raise
        print(f"吴语语音生成失败，改用普通话兜底语音：{exc}")
        fallback = output_dir / f"turn_{turn:03d}_reply_mandarin_fallback.mp3"
        return synthesize_mp3(
            TTSRequest(
                text=text,
                output_path=fallback,
                voice=args.tts_voice,
                rate=args.tts_rate,
                pitch=args.tts_pitch,
                backend="edge",
                command_template=args.tts_command,
            )
        )


def _append_json_log(path: str | None, row: dict) -> None:
    if not path:
        return
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _recording_payload(recording) -> dict | None:
    if recording is None:
        return None
    return {
        "path": str(recording.path),
        "duration_seconds": recording.duration_seconds,
        "speech_started": recording.speech_started,
        "peak_rms": recording.peak_rms,
    }


def _write_turn_report(output_dir: Path, turn: int, report: dict) -> Path:
    path = output_dir / f"turn_{turn:03d}_report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _print_input_devices() -> None:
    rows = list_input_devices()
    if not rows:
        print("没有发现可用麦克风输入设备。")
        return
    print("可用麦克风输入设备：")
    for row in rows:
        print(
            f"[{row['index']}] {row['name']} "
            f"(channels={row['max_input_channels']}, default_sr={row['default_samplerate']:.0f})"
        )


def _parse_input_device(value: str | None) -> int | str | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return value


if __name__ == "__main__":
    raise SystemExit(main())
