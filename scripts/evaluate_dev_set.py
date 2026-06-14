from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Iterable


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the 92-item Shanghainese validation split.")
    parser.add_argument("--manifest", default="data/splits/dev.jsonl")
    parser.add_argument("--model", default="outputs/models/whisper-small-shanghai-lora-full")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--name", default="dev_eval_92")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=225)
    parser.add_argument("--num-beams", type=int, default=1)
    args = parser.parse_args()

    try:
        import torch
        from peft import PeftConfig, PeftModel
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
    except ImportError as exc:
        raise RuntimeError("Install ASR/fine-tune dependencies first.") from exc

    rows = load_manifest(Path(args.manifest))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(args.model)
    if not (model_path / "adapter_config.json").exists():
        raise ValueError(f"Expected a LoRA adapter directory: {model_path}")

    peft_config = PeftConfig.from_pretrained(str(model_path))
    base_model = peft_config.base_model_name_or_path or "TingChen-ppmc/whisper-small-Shanghai"
    bundled_base = Path("models/whisper-small-Shanghai")
    if bundled_base.exists():
        base_model = str(bundled_base)

    processor_name = str(model_path) if (model_path / "preprocessor_config.json").exists() else base_model
    processor = AutoProcessor.from_pretrained(processor_name, local_files_only=args.local_files_only)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(base_model, local_files_only=args.local_files_only)
    model = PeftModel.from_pretrained(model, str(model_path), is_trainable=False)
    if hasattr(model, "merge_and_unload"):
        model = model.merge_and_unload()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    if hasattr(model, "generation_config"):
        model.generation_config.language = "Chinese"
        model.generation_config.task = "transcribe"
        model.generation_config.forced_decoder_ids = None
        model.generation_config.suppress_tokens = []

    results: list[dict] = []
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        audio_arrays = [load_audio_array(item["audio"], target_sampling_rate=16000) for item in batch]
        features = processor.feature_extractor(
            audio_arrays,
            sampling_rate=16000,
            return_tensors="pt",
            return_attention_mask=True,
        )
        input_features = features.input_features.to(device)
        attention_mask = getattr(features, "attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        with torch.inference_mode():
            generated = model.generate(
                input_features,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                language="Chinese",
                task="transcribe",
                num_beams=args.num_beams,
            )
        predictions = processor.tokenizer.batch_decode(generated, skip_special_tokens=True)
        for item, prediction in zip(batch, predictions):
            reference = item["text"]
            hyp = prediction.strip()
            distance = levenshtein_distance(normalize_text(reference), normalize_text(hyp))
            ref_len = max(len(normalize_text(reference)), 1)
            results.append(
                {
                    "audio": item["audio"],
                    "speaker_id": item.get("speaker_id"),
                    "gender": item.get("gender"),
                    "reference": reference,
                    "prediction": hyp,
                    "char_distance": distance,
                    "reference_chars": ref_len,
                    "cer": distance / ref_len,
                }
            )
        print(f"Evaluated {min(start + len(batch), len(rows))}/{len(rows)}")

    summary = summarize(results)
    prefix = output_dir / args.name
    (prefix.with_suffix(".predictions.jsonl")).write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in results) + "\n",
        encoding="utf-8",
    )
    (prefix.with_suffix(".summary.json")).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (prefix.with_suffix(".report.md")).write_text(
        render_report(summary, results),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def load_manifest(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("audio") and row.get("text"):
                rows.append(row)
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def load_audio_array(path: str, target_sampling_rate: int):
    import numpy as np
    import soundfile as sf

    array, sampling_rate = sf.read(path, dtype="float32")
    if getattr(array, "ndim", 1) > 1:
        array = array.mean(axis=1)
    if sampling_rate != target_sampling_rate:
        old_positions = np.linspace(0.0, 1.0, num=len(array), endpoint=False)
        new_length = max(1, round(len(array) * target_sampling_rate / sampling_rate))
        new_positions = np.linspace(0.0, 1.0, num=new_length, endpoint=False)
        array = np.interp(new_positions, old_positions, array).astype("float32")
    return array


def normalize_text(text: str) -> str:
    punctuation = set(" \t\r\n，。！？；：、,.!?;:\"'“”‘’（）()[]{}<>《》")
    return "".join(char.lower() for char in text if char not in punctuation)


def levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            current.append(
                min(
                    current[j - 1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (0 if char_a == char_b else 1),
                )
            )
        previous = current
    return previous[-1]


@dataclass
class GroupStats:
    count: int
    corpus_cer: float
    mean_item_cer: float


def group_stats(rows: Iterable[dict], key: str) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key) or "unknown")].append(row)
    return {name: compute_group_stats(items).__dict__ for name, items in sorted(grouped.items())}


def compute_group_stats(rows: list[dict]) -> GroupStats:
    distance = sum(int(row["char_distance"]) for row in rows)
    ref_chars = sum(int(row["reference_chars"]) for row in rows)
    return GroupStats(
        count=len(rows),
        corpus_cer=round(distance / max(ref_chars, 1), 6),
        mean_item_cer=round(mean(float(row["cer"]) for row in rows), 6),
    )


def summarize(results: list[dict]) -> dict:
    sorted_by_error = sorted(results, key=lambda row: row["cer"], reverse=True)
    distances = sum(int(row["char_distance"]) for row in results)
    ref_chars = sum(int(row["reference_chars"]) for row in results)
    item_cers = sorted(float(row["cer"]) for row in results)
    exact_matches = sum(1 for row in results if normalize_text(row["reference"]) == normalize_text(row["prediction"]))
    return {
        "sample_count": len(results),
        "corpus_cer": round(distances / max(ref_chars, 1), 6),
        "mean_item_cer": round(mean(item_cers), 6),
        "median_item_cer": round(median(item_cers), 6),
        "p90_item_cer": round(percentile(item_cers, 0.9), 6),
        "max_item_cer": round(item_cers[-1], 6),
        "exact_match_rate": round(exact_matches / max(len(results), 1), 6),
        "total_char_distance": distances,
        "total_reference_chars": ref_chars,
        "by_gender": group_stats(results, "gender"),
        "by_speaker": group_stats(results, "speaker_id"),
        "worst_examples": [
            {
                "audio": row["audio"],
                "speaker_id": row.get("speaker_id"),
                "gender": row.get("gender"),
                "cer": round(float(row["cer"]), 6),
                "reference": row["reference"],
                "prediction": row["prediction"],
            }
            for row in sorted_by_error[:10]
        ],
    }


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    index = min(len(values) - 1, max(0, round((len(values) - 1) * q)))
    return values[index]


def render_report(summary: dict, results: list[dict]) -> str:
    quality = "good" if summary["corpus_cer"] <= 0.15 else "needs_improvement"
    lines = [
        "# 92 条验证集评估报告",
        "",
        f"- 样本数: {summary['sample_count']}",
        f"- Corpus CER: {summary['corpus_cer']:.4f}",
        f"- 单条平均 CER: {summary['mean_item_cer']:.4f}",
        f"- 单条中位 CER: {summary['median_item_cer']:.4f}",
        f"- 单条 P90 CER: {summary['p90_item_cer']:.4f}",
        f"- 完全匹配率: {summary['exact_match_rate']:.4f}",
        f"- 质量判断: {quality}",
        "",
        "## 可完善点",
    ]
    if summary["corpus_cer"] <= 0.15:
        lines.append("- 总体结果可以作为课程 agent 使用，但最差样例还有明显优化空间。")
    else:
        lines.append("- 总体结果不够稳，提交前应优先处理下面几类问题。")
    lines.extend(
        [
            "- 人工复核最差 CER 的音频，检查是否存在标注不一致、噪声、截断、过短句子或同音异写。",
            "- 固定当前 3700/92 划分继续调参，避免换验证集造成指标不可比。",
            "- 训练侧可尝试 5 epochs + early stopping、学习率 5e-5、LoRA rank 32，观察是否能改善高 CER 说话人。",
            "- 解码侧可比较 greedy 与 beam search；如果 beam search 变好，可把 UI/CLI 默认解码参数改掉。",
            "- 对常见上海话异写词做规范化，例如同音功能词、语气词、口语连读词，分别用于评分和展示。",
            "- 高错误说话人需要做 speaker-balanced 验证；若集中在少数 speaker，可做速度扰动、轻噪声增强或补充相近说话人样本。",
            "",
            "## 最差样例",
        ]
    )
    for idx, row in enumerate(summary["worst_examples"], start=1):
        lines.extend(
            [
                f"### {idx}. {row['audio']} CER={row['cer']:.4f}",
                f"- 说话人: {row.get('speaker_id')} / {row.get('gender')}",
                f"- 标注: {row['reference']}",
                f"- 预测: {row['prediction']}",
                "",
            ]
        )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
