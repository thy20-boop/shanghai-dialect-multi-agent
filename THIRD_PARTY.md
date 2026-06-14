# 第三方模型与数据来源

本仓库只保存项目代码、小型参考样例和评估结果，不重新分发大型模型权重。

## WenetSpeech-Wu

- 项目与数据说明：https://github.com/ASLP-lab/WenetSpeech-Wu-Repo
- 语音理解模型：https://huggingface.co/ASLP-lab/WenetSpeech-Wu-Speech-Understanding
- 语音生成模型：https://huggingface.co/ASLP-lab/WenetSpeech-Wu-Speech-Generation

## CosyVoice

- 官方代码：https://github.com/FunAudioLLM/CosyVoice

## Dolphin

- 本项目通过 `dataoceanai-dolphin` Python 包调用 Dolphin 方言 ASR。

## 上海话会话数据

- 数据集：`TingChen-ppmc/Shanghai_Dialect_Conversational_Speech_Corpus`
- 仓库中的 `data/splits/` 只保存训练/验证划分元数据。
- `assets/reference/official_shanghai_prompt.wav` 是用于复现实验的短参考片段，对应文本为：
  `最少辰光阿拉是做撒呃喃，有钞票就是到银行里保本保息。`

使用、展示或再分发上述资源时，请同时遵守各上游项目页面列出的许可证和使用条件。
