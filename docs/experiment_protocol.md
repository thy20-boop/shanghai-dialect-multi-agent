# 上海话实验流程

## 1. 准备数据

```powershell
python scripts\fetch_shanghai_hf_dataset.py `
  --output data\shanghai_manifest.jsonl `
  --audio-dir data\shanghai_audio

python scripts\split_manifest.py `
  --manifest data\shanghai_manifest.jsonl `
  --output-dir data\splits
```

## 2. 跑 baseline

```powershell
python -m ganagent.cli batch `
  --manifest data\splits\test.jsonl `
  --output outputs\baseline_predictions.jsonl `
  --backend whisper `
  --model openai/whisper-small
```

上海话微调 baseline：

```powershell
python -m ganagent.cli batch `
  --manifest data\splits\test.jsonl `
  --output outputs\shanghai_baseline_predictions.jsonl `
  --backend whisper `
  --model TingChen-ppmc/whisper-small-Shanghai
```

## 3. LoRA

```powershell
python scripts\finetune_whisper_lora.py `
  --manifest data\splits\train.jsonl `
  --model openai/whisper-small `
  --output-dir outputs\whisper-small-shanghai-lora `
  --max-train-samples 500
```

## 4. 评估

```powershell
python -m ganagent.cli evaluate --predictions outputs\baseline_predictions.jsonl
python -m ganagent.cli evaluate --predictions outputs\shanghai_baseline_predictions.jsonl
```

重点比较 `阿拉`、`侬`、`吾`、`搿个`、`勿`、`伐` 等方言词，以及 LoRA、CER、Whisper 等领域词。
