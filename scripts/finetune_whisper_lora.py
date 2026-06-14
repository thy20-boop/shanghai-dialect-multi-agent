from __future__ import annotations

import argparse
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tune Whisper with LoRA for Shanghainese ASR.")
    parser.add_argument("--manifest", default=None, help="Backward-compatible single manifest input.")
    parser.add_argument("--train-manifest", default=None, help="JSONL with 3700 training examples.")
    parser.add_argument("--eval-manifest", default=None, help="JSONL with 92 validation examples.")
    parser.add_argument("--model", default="TingChen-ppmc/whisper-small-Shanghai")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--max-steps", type=int, default=-1, help="Override epochs with an exact training step budget.")
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    args = parser.parse_args()

    try:
        import numpy as np
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model
        from transformers import (
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
            WhisperForConditionalGeneration,
            WhisperProcessor,
            set_seed,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Install fine-tune dependencies first: python -m pip install -e .[finetune]"
        ) from exc

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows, eval_rows = load_training_rows(args)
    train_dataset = Dataset.from_list(train_rows)
    eval_dataset = Dataset.from_list(eval_rows)

    processor = WhisperProcessor.from_pretrained(
        args.model,
        language="Chinese",
        task="transcribe",
        local_files_only=args.local_files_only,
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
    )
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False
    if hasattr(model, "generation_config"):
        model.generation_config.language = "Chinese"
        model.generation_config.task = "transcribe"
        model.generation_config.forced_decoder_ids = None
        model.generation_config.suppress_tokens = []

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    if not args.no_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    model.print_trainable_parameters()

    def prepare_example(batch: dict[str, Any]) -> dict[str, Any]:
        audio_array, sampling_rate = load_audio_array(batch["audio"], target_sampling_rate=16000)
        features = processor.feature_extractor(
            audio_array,
            sampling_rate=sampling_rate,
            return_attention_mask=False,
        )
        batch["input_features"] = features.input_features[0]
        batch["labels"] = processor.tokenizer(batch["text"]).input_ids
        return batch

    prepared_train = train_dataset.map(
        prepare_example,
        remove_columns=train_dataset.column_names,
        desc="Preparing train audio",
    )
    prepared_eval = eval_dataset.map(
        prepare_example,
        remove_columns=eval_dataset.column_names,
        desc="Preparing validation audio",
    )

    training_kwargs = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "warmup_steps": args.warmup_steps,
        "num_train_epochs": args.num_train_epochs,
        "max_steps": args.max_steps,
        "fp16": torch.cuda.is_available(),
        "predict_with_generate": True,
        "generation_max_length": 225,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "save_strategy": "steps",
        "report_to": [],
        "remove_unused_columns": False,
        "save_total_limit": args.save_total_limit,
        "load_best_model_at_end": True,
        "metric_for_best_model": "cer",
        "greater_is_better": False,
        "seed": args.seed,
    }
    signature = inspect.signature(Seq2SeqTrainingArguments)
    if "eval_strategy" in signature.parameters:
        training_kwargs["eval_strategy"] = "steps"
    else:
        training_kwargs["evaluation_strategy"] = "steps"
    training_args = Seq2SeqTrainingArguments(**training_kwargs)

    def compute_metrics(pred: Any) -> dict[str, float]:
        import jiwer

        predictions = pred.predictions
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        label_ids = pred.label_ids
        label_ids = np.where(label_ids == -100, processor.tokenizer.pad_token_id, label_ids)
        pred_str = processor.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        return {"cer": float(jiwer.cer(label_str, pred_str))}

    trainer_kwargs = {
        "args": training_args,
        "model": model,
        "train_dataset": prepared_train,
        "eval_dataset": prepared_eval,
        "data_collator": WhisperDataCollator(processor=processor),
        "compute_metrics": compute_metrics,
    }
    trainer_signature = inspect.signature(Seq2SeqTrainer)
    if "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = processor
    else:
        trainer_kwargs["tokenizer"] = processor
    trainer = Seq2SeqTrainer(**trainer_kwargs)

    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))
    metrics = trainer.evaluate()
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_metrics("eval", metrics)
    metadata = {
        "base_model": args.model,
        "train_manifest": args.train_manifest,
        "eval_manifest": args.eval_manifest,
        "train_examples": len(train_rows),
        "eval_examples": len(eval_rows),
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "max_steps": args.max_steps,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "save_total_limit": args.save_total_limit,
        "cuda_available": bool(torch.cuda.is_available()),
        "best_metric": trainer.state.best_metric,
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "final_eval": metrics,
    }
    (output_dir / "training_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved LoRA adapter, processor, and training metadata to {output_dir}")
    return 0


def load_training_rows(args: argparse.Namespace) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if args.train_manifest and args.eval_manifest:
        train_rows = load_manifest(Path(args.train_manifest), args.max_train_samples)
        eval_rows = load_manifest(Path(args.eval_manifest), None)
        return train_rows, eval_rows

    if not args.manifest:
        raise ValueError("Provide --train-manifest and --eval-manifest for full training.")

    try:
        from datasets import Dataset
    except ImportError as exc:
        raise RuntimeError("Install datasets first.") from exc

    rows = load_manifest(Path(args.manifest), args.max_train_samples)
    dataset = Dataset.from_list(rows)
    split = (
        dataset.train_test_split(test_size=args.eval_ratio, seed=args.seed)
        if len(dataset) > 1
        else {"train": dataset, "test": dataset}
    )
    return list(split["train"]), list(split["test"])


def load_manifest(path: Path, limit: int | None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("audio") and item.get("text"):
                rows.append({"audio": item["audio"], "text": item["text"]})
            if limit is not None and len(rows) >= limit:
                break
    if not rows:
        raise ValueError(f"No usable examples found in {path}")
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
        sampling_rate = target_sampling_rate
    return array, sampling_rate


@dataclass
class WhisperDataCollator:
    processor: Any

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        bos_token_id = self.processor.tokenizer.bos_token_id
        if labels.shape[1] > 0 and (labels[:, 0] == bos_token_id).all().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


if __name__ == "__main__":
    raise SystemExit(main())
