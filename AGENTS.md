# Shanghai Dialect Codex Agent

本项目是一个 Codex-native 多智能体课程作业。不要把“联网回答用户问题”做成 Streamlit 网页功能，也不要要求 OpenAI API key；联网搜索和答案整理由 Codex 当前会话完成。

## 默认工作流

当用户给出上海话/吴语音频并要求回答问题时：

1. 先运行识别与翻译：

```powershell
.\.venv\Scripts\python.exe -m ganagent.cli translate --audio <audio-path> --codex-task-output outputs\codex_question_task.md --no-save-active-learning
```

2. 读取 `outputs\codex_question_task.md`，把“用户问题候选”当成待确认问题。
3. 如果识别质量低、问题不完整或可疑片段影响语义，先说明不确定点；必要时请用户确认。
4. 对新闻、地点、政策、价格、人物、产品、课程资料等可能变化的信息，必须由 Codex 使用联网搜索后再回答。
5. 回答用中文，保留来源链接；如果证据不足，明确说明“目前没有足够证据”。
6. 把最终回答写入 `outputs\codex_answer.txt`，并默认同时生成普通话 MP3、吴语口语稿文本和吴语口语稿 MP3：

```powershell
.\.venv\Scripts\python.exe -m ganagent.cli speak --text-file outputs\codex_answer.txt --output outputs\codex_answer.mp3 --wu-output outputs\codex_answer_wu.mp3 --wu-text-output outputs\codex_answer_wu.txt
```

普通话 MP3 给一般用户听；吴语 MP3 由 WenetSpeech-Wu `CosyVoice2-Wu-SFT` 生成，并由 Whisper-Medium-Wu 回听检查。

首次运行吴语生成专家：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_wenet_wu_sft.ps1
powershell -ExecutionPolicy Bypass -File scripts\start_wenet_wu_expert.ps1 -Detached
```

高风险回答使用回听仲裁：

```powershell
.\.venv\Scripts\python.exe -m ganagent.cli speak-verified --text-file outputs\codex_answer.txt --target wuu --output outputs\codex_answer_wu.mp3 --verify-backend whisper_medium_wu
```

## 智能体分工

- 音频预处理智能体：加载音频/视频并统一采样。
- Dolphin 主识别智能体：做上海话/中文方言主 ASR。
- 本地 LoRA 复核智能体：保留自研模型价值，参与候选复核。
- 候选仲裁智能体：处理长视频重复、短句混淆和候选分歧。
- 纠错记忆智能体：应用术语表、地名、人名和用户纠错记忆。
- 风险检测智能体：输出质量评分、可疑片段和主动学习候选。
- Codex 联网回答智能体：在本会话中搜索网页、核验时效信息、生成答案和来源。
- 语音输出智能体：普通话使用 edge-tts；吴语使用 CosyVoice2-Wu，并由参考专家、风险检测和候选仲裁选择最终音频。

## 验证命令

```powershell
.\.venv\Scripts\python.exe -m compileall src app tests
.\.venv\Scripts\python.exe -m pytest
```
