# Shanghai Dialect ASR Repair Agent

这是一个面向上海话/吴语的多智能体语音理解与普通话转换 agent。当前版本已经使用公开上海话会话语音数据做正式训练，同时引入 Dolphin 方言 ASR 作为强识别底座，并用本项目自己的候选仲裁、纠错记忆、风险检测和主动学习闭环补齐它的短板。

## 当前状态

- 数据集：`TingChen-ppmc/Shanghai_Dialect_Conversational_Speech_Corpus`
- 全量音频：3792 条
- 训练集：3700 条，`data/splits/train.jsonl`
- 验证集：92 条，`data/splits/dev.jsonl`
- 自训练模型产物（本机）：`outputs/models/whisper-small-shanghai-lora-full`，不随 GitHub 重复分发
- 默认识别后端：`dolphin_multiagent`
- 主识别模型：`Dolphin small.cn` 中文方言模型，上海地区提示 + deep-biasing 热词
- 复核/自研模型：本地训练 Whisper-LoRA，用于第二候选、兜底和主动学习对照
- 多智能体模块：音频预处理、Dolphin ASR、官方 Whisper-Medium-Wu 复核、候选仲裁、纠错记忆、普通话转换、风险检测、主动学习
- 主动学习队列：`data/active_learning_queue.jsonl`
- 质量控制：输出质量评分、候选共识评分和下一步处理建议
- 语音输出：默认使用 WenetSpeech-Wu CosyVoice2 吴语专家，并由双 ASR 回识、风险检测和候选仲裁选择最终 MP3
- Codex 联网回答：可把识别内容整理成 Codex 问答任务，由 Codex 当前会话联网搜索并生成带来源回答
- Whisper 备用解码：beam=5
- 本地 LoRA 在 92 条验证集 beam=5 Corpus CER：0.1107

## 直接运行

双击：

```text
START_UI.bat
```

浏览器打开：

```text
http://localhost:8501
```

## 实时对话 Agent

如果不想上传 MP3/视频，可以直接启动按轮实时对话版：

```text
双击 START_LIVE_AGENT.bat
```

或者命令行运行：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-live.txt
.\.venv\Scripts\python.exe -m ganagent.live_agent --reply-target wuu
```

启动后直接对着麦克风说上海话或普通话，停顿约 1 秒后系统会自动结束本轮录音，并执行：

```text
麦克风输入 -> VAD 断句 -> 多智能体识别/纠错/风险检测 -> 对话管理 -> 吴语/普通话语音回复
```

实时版的本地回答覆盖常见社区服务问题，例如身份证遗失、居住证办理、报警、火警、急救和 12345 热线。涉及天气、新闻、政策、营业时间、地址电话等可能变化的问题时，系统会在 `outputs/live_agent/` 生成 Codex 联网任务文件，由 Codex 当前会话继续搜索和核对来源。

常用参数：

```powershell
.\.venv\Scripts\python.exe -m ganagent.live_agent --turns 3
.\.venv\Scripts\python.exe -m ganagent.live_agent --reply-target mandarin
.\.venv\Scripts\python.exe -m ganagent.live_agent --no-playback
.\.venv\Scripts\python.exe -m ganagent.live_agent --start-threshold 0.018 --silence-seconds 1.1
```

如果吴语 TTS 服务未启动，实时版会自动退回普通话语音，避免课堂演示时中断。`START_LIVE_AGENT.bat` 会先尝试启动 WenetSpeech-Wu 专家服务。

支持上传 `WAV / FLAC / OGG / M4A / MP3 / MP4 / AAC`。如果文件后缀和真实音频封装不一致，例如 M4A 被改名成 `.flac`，程序会自动用 ffmpeg 兜底解码。

高级设置里可以勾选“生成语音 MP3”。朗读内容支持：

- `普通话结果`：把最终普通话转换结果合成为 MP3。
- `吴语语音`：CosyVoice2-Wu 直接接收清晰的语义文本，由吴语声学模型产生上海话发音；只有非吴语声学后端才使用词典口语改写。

普通话输出仍使用 `edge-tts`。吴语输出默认使用 `CosyVoice2-Wu-SFT` 专家；运行目录优先读取环境变量 `SHANGHAI_WU_RUNTIME`，兼容已有的 `D:\wswu_runtime`，新电脑默认使用 `%LOCALAPPDATA%\ShanghaiDialectAgent\wswu_runtime`。`START_UI.bat` 会自动安装并启动本地服务。也可以单独安装和启动：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_wenet_wu_sft.ps1
powershell -ExecutionPolicy Bypass -File scripts\start_wenet_wu_expert.ps1 -Detached
```

完整的吴语回答链路由本项目自己的 `ShanghaiGuard-Wu` 生成专家控制层管理：它把 CosyVoice2-Wu-SFT 作为公开声学底座，但自己负责任务风险识别、参考音频专家路由、固定 seed/语速候选计划、Whisper-Medium-Wu 回听、关键实体硬门槛、句首赘词裁剪和候选仲裁。普通政务句可在男声、女声参考专家之间竞争；电话号码等高风险回答只使用认证参考且禁用裁剪。电话、身份证等关键实体必须完整匹配，最终片段之间加入 0.55 秒停顿后再输出 MP3。可通过 `--secondary-cosyvoice-wu-url` 接入 API 兼容的第二生成专家，而不替换当前 Wu-SFT。

`ShanghaiGuard-Wu` 的设计说明见 [docs/shanghaiguard_wu_generation_expert.md](docs/shanghaiguard_wu_generation_expert.md)。项目没有把开源模型冒充成自训 TTS 大模型；自己的技术贡献集中在多智能体生成控制、回听验收和风险约束层。

本次 WenetSpeech-Wu 专家对照、部署修复和两段最终评估见 [docs/WENETSPEECH_WU_DEPLOYMENT_REPORT.md](docs/WENETSPEECH_WU_DEPLOYMENT_REPORT.md)。

```powershell
.\.venv\Scripts\python.exe -m ganagent.cli speak-verified `
  --target wuu `
  --text-file outputs\codex_answer.txt `
  --output outputs\codex_answer_wu.mp3 `
  --tts-backend cosyvoice_wu `
  --verify-backend dolphin_multiagent
```

部署结构和验证结果见 `docs/wenetspeech_wu_expert.md`。

联网回答不放在网页里，也不需要 OpenAI API key。项目根目录的 `AGENTS.md` 定义了 Codex-native 工作流：程序先识别上海话并生成 `outputs/codex_question_task.md`，然后由 Codex 当前会话搜索网页、核验来源、回答问题。这样“联网搜索智能体”就是 Codex 本身，符合课程里的多智能体协同设定。

当前默认使用多智能体协同模式：Dolphin 先做上海话主识别，官方 WenetSpeech-Wu `Whisper-Medium-Wu` 作为吴语复核/兜底候选。Dolphin 会使用上海地区提示，并带常见上海话热词，例如 `先生、侬好、初次见面、第一趟、阿拉、拧来`。最终结果仍经过本项目自己的候选仲裁、纠错记忆、上海话词典、可疑片段检测、普通话转换和主动学习逻辑。

吴语语音回答采用逐句生成和逐句回识。电话号码、身份证、派出所、户籍等关键实体实行 100% 硬门槛；任一实体缺失时只重试对应句，三次仍失败则拒绝发布该吴语音频并使用清晰兜底语音。

首次启动会运行 `scripts/setup_whisper_medium_wu.ps1`，下载约 3.27 GB 的官方检查点到统一吴语运行目录。

如果切换到 Whisper 后端，长视频/长音频会先按音轨停顿自动切成短语音片段，默认最长 8 秒，再逐段识别；没有稳定停顿时才回退到固定切片。Whisper 备用链路开启 beam=5、`condition_on_prev_tokens=False` 和 `no_repeat_ngram_size=6`，并会把偶发的长重复幻觉压缩成 `[重复片段省略]`。

高级设置里可以切换识别后端或开源候选，用来做 A/B 对比：

- `Whisper-Wu`：公共吴语 Whisper/LoRA 候选，模型填 `peft:kaiwang0574/whisper-wu`
- `FunASR/SenseVoice`：工程成熟的中文/多方言开源 ASR 候选
- `Dolphin`：DataoceanAI 的中文方言 ASR 候选

这些候选不是课程项目的全部内容，而是多智能体系统里的不同技能来源。首次使用前需要联网并额外安装：

```powershell
python -m pip install -r requirements-experimental-asr.txt
```

GitHub 课程提交版不包含本地训练权重和公开模型缓存；Dolphin 与官方吴语专家首次使用需要联网下载，缓存后可以离线运行。

如果要只看 Dolphin 单模型效果，可以在网页高级设置选择 `dolphin` 后端；如果要只看官方吴语专家，可以选择 `whisper_medium_wu`。默认 `dolphin_multiagent` 会把 Dolphin 主输出和 Whisper-Medium-Wu 复核结果同时保留下来；候选仲裁智能体只有在主输出出现重复幻觉、乱码、极短空结果或已知短句混淆等明显风险时，才会通过 `alternative_rerank` 接管最终输出。

命令行示例：

```powershell
python -m ganagent.cli translate `
  --audio path\to\audio.wav `
  --json
```

可通过环境变量覆盖 Dolphin 热词：

```powershell
$env:SHANGHAI_DOLPHIN_HOTWORDS="先生,侬好,初次见面,第一趟,上海"
```

## 命令行调用

单条音频转普通话：

```powershell
.\scripts\translate_audio.ps1 -Audio path\to\audio.m4a -Json
```

生成吴语口语稿 MP3：

```powershell
.\scripts\translate_audio.ps1 `
  -Audio path\to\audio.m4a `
  -TtsOutput outputs\wu_output.mp3 `
  -TtsTarget wuu `
  -Online
```

或者：

```powershell
python -m ganagent.cli translate `
  --audio path\to\audio.m4a `
  --backend whisper `
  --model outputs\models\whisper-small-shanghai-lora-full `
  --max-speech-region-seconds 8 `
  --json
```

命令行直接生成 MP3：

```powershell
python -m ganagent.cli translate `
  --audio path\to\audio.m4a `
  --tts-output outputs\wu_output.mp3 `
  --tts-target wuu
```

生成 Codex 联网问答任务：

```powershell
python -m ganagent.cli translate `
  --audio path\to\question.wav `
  --codex-task-output outputs\codex_question_task.md
```

然后把 `outputs/codex_question_task.md` 交给 Codex 读取并联网回答。回答完成后，把答案保存到 `outputs/codex_answer.txt`，一条命令同时生成普通话 MP3、训练集风格吴语口语稿和吴语口语稿 MP3：

```powershell
python -m ganagent.cli speak `
  --text-file outputs\codex_answer.txt `
  --output outputs\codex_answer.mp3 `
  --wu-output outputs\codex_answer_wu.mp3 `
  --wu-text-output outputs\codex_answer_wu.txt
```

也可以只生成吴语口语稿 MP3：

```powershell
python -m ganagent.cli speak `
  --text-file outputs\codex_answer.txt `
  --target wuu `
  --output outputs\codex_answer_wu.mp3
```

长视频建议保持默认的停顿切分。如果要调参，可以把 `--max-speech-region-seconds` 调到 `5-8`；如果要退回旧的固定切片方式，可加 `--no-vad`。

如果某段视频里有人名、地名、店名或专有词总是识别错，可以在网页高级设置的“补充修复词”里临时加入：

```text
错词=正确词
车子机面=初次见面
王家=王佳
```

如果这个纠错要长期保留，写进 `data/user_corrections.json`：

```json
{
  "王家": "王佳",
  "车子机面": "初次见面"
}
```

命令行也支持重复传入：

```powershell
python -m ganagent.cli translate `
  --audio path\to\video.mp4 `
  --backend whisper `
  --custom-repair "车子机面=初次见面" `
  --custom-repair "王家=王佳"
```

长期纠错记忆也可以从命令行指定：

```powershell
python -m ganagent.cli translate `
  --audio path\to\video.mp4 `
  --memory data\user_corrections.json `
  --json
```

拖拽音频到：

```text
TRANSCRIBE_AUDIO.bat
```

## 输出说明

网页会显示：

- 普通话结果
- 识别原文
- 识别状态
- 质量评分
- 修复次数
- 可疑片段
- 修复记录
- 多智能体协作记录
- 主动学习候选
- 完整 JSON

`修复记录` 包含几类：

- `asr_repair`：识别文本纠错，例如技术词、错别字、上下文词修复
- `alternative_rerank`：主模型明显不稳时，用 Dolphin/LoRA 等候选结果接管当前输出
- `dialect_translation`：上海话词转普通话，例如 `阿拉 -> 我们`、`拧来 -> 人来`

`可疑片段` 包含低置信度、重复、乱码、技术词混杂，以及普通话结果里可能残留的上海话词。

`多智能体协作记录` 会列出每个智能体的角色、状态和输出摘要，方便答辩时解释系统并不是简单调用某一个模型。

`主动学习候选` 会把高风险片段、候选分歧和明显修复样本写入 `data/active_learning_queue.jsonl`。后续人工确认正确文本后，可以写入 `data/user_corrections.json` 或加入下一轮 LoRA 微调清单。

## 主动学习闭环

查看主动学习队列概览：

```powershell
python -m ganagent.cli learning --json
```

生成 Markdown 报告：

```powershell
python -m ganagent.cli learning `
  --report outputs\active_learning_report.md
```

如果已经人工确认了队列项，在 JSONL 里给样本补上 `confirmed_transcript` 或 `confirmed_text`，并把 `status` 改成 `confirmed`，即可导出下一轮训练 manifest：

```powershell
python -m ganagent.cli learning `
  --export-manifest data\splits\active_learning_confirmed.jsonl
```

未确认样本默认不会进入训练集，避免把模型自己的错误再次喂回模型。

## 验证报告

验证集结果文件：

```text
outputs/dev_eval_92_comparison.md
outputs/dev_eval_92_beam5.report.md
outputs/dev_eval_92_beam5.summary.json
outputs/dev_eval_92_beam5.predictions.jsonl
```

V2 多智能体迭代说明：

```text
docs/v2_iteration_report.md
```

## 重新训练

固定 3700/92 划分重新训练：

```powershell
.\scripts\install_deps.ps1 -Group finetune -TorchBackend cuda
python scripts\finetune_whisper_lora.py `
  --train-manifest data\splits\train.jsonl `
  --eval-manifest data\splits\dev.jsonl `
  --model TingChen-ppmc/whisper-small-Shanghai `
  --output-dir outputs\models\whisper-small-shanghai-lora-full
```

## 打包

```powershell
powershell -ExecutionPolicy Bypass -File scripts\package_online.ps1 -Version full-trained
powershell -ExecutionPolicy Bypass -File scripts\package_release.ps1 -Version full-trained
powershell -ExecutionPolicy Bypass -File scripts\package_portable.ps1 -Version full-trained
```

输出在：

```text
outputs/releases/
```

## 目录

```text
app/                 Streamlit 网页界面
src/ganagent/        agent 核心逻辑
data/examples/       上海话词典和示例 manifest
data/splits/         3700/92 固定划分
scripts/             下载、训练、验证、打包脚本
outputs/models/      本地训练产物（GitHub 默认忽略）
outputs/releases/    最终交付包
tests/               单元测试
```
