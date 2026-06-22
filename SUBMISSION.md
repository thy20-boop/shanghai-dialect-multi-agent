# 课程作业提交说明

## 项目题目

上海话多智能体语音问答 Agent：识别、纠错、风险检测、联网回答与吴语语音生成。

## 核心贡献

本项目不是简单封装单个开源模型，而是将多个专家组织成可检查、可回退的协作链路：

1. Dolphin 负责方言主识别。
2. Whisper-Medium-Wu 负责吴语复核。
3. 候选仲裁智能体综合多个转写结果。
4. 自动纠错智能体结合领域词典和持久化纠错记忆。
5. 风险检测智能体标记低置信度、重复幻觉和关键实体缺失。
6. Codex 任务生成模块把用户问题交给 Codex 搜索并组织回答。
7. CosyVoice2-Wu 生成上海话语音回答。
8. 吴语回听评估智能体检查 CER、关键词和电话号码等关键实体。
9. 参考音频专家、句首幻觉裁剪与候选仲裁共同选择最终 MP3。
10. `ShanghaiGuard-Wu` 是本项目自己的吴语生成专家控制层：公开 CosyVoice2-Wu-SFT 负责声学生成，本项目负责风险识别、参考专家路由、候选计划、回听验收、关键实体硬门槛和最终仲裁。
11. 实时对话 Agent 支持麦克风输入、VAD 自动断句、多智能体识别、本地对话回答、Codex 联网任务生成和语音播放，不再要求用户先上传 MP3。

## 我们自己的技术含量

最终吴语 MP3 并不是直接发布开源模型的第一条输出。项目新增了 `src/ganagent/wu_generation_expert.py`，把吴语生成拆成可解释、可复现的多智能体流程：

- 任务风险分类：区分 `hotline`、`public_service`、`general`，热线/电话类自动进入最高风险策略。
- 参考专家路由：普通政务句允许多说话人参考竞争，热线类只允许带 `hotline` 认证标签的参考。
- 确定性候选计划：固定 seed 和语速生成多条候选，方便复现实验和答辩展示。
- 回听自验证：每条候选用识别模型回听，计算关键词召回、字符准确率、方言信号和可疑片段。
- 关键实体硬门槛：电话、身份证、派出所、户籍等实体必须完整保留。
- 保守裁剪：只在非高风险任务中裁剪句首赘词，并且裁剪后必须二次回听分数更高才采用。
- 实时交互：`src/ganagent/live_agent.py` 将原来的文件上传流程包装成按轮语音对话，`src/ganagent/audio_capture.py` 负责麦克风采集和播放，`src/ganagent/dialogue_manager.py` 负责本地社区服务问答与 Codex 搜索任务分流。

详细说明见 `docs/shanghaiguard_wu_generation_expert.md`。

## 可复现结果

| 测试回答 | 整段字符准确率 | 关键实体召回 |
| --- | ---: | ---: |
| 身份证求助 | 85.87% | 100% |
| 紧急求助电话 | 76.64% | 100% |

测试命令：

```powershell
.venv\Scripts\python.exe -m pytest -q
```

当前版本共有 72 项自动化测试。

## 仓库内容

- `src/ganagent/`：多智能体核心实现。
- `app/`：Streamlit 演示界面。
- `scripts/`：模型安装、训练、评估和打包脚本。
- `configs/`：智能体与参考音频专家配置。
- `data/splits/`：3700 条训练、92 条验证的固定划分元数据。
- `docs/`：架构、实验和 PPT 介绍材料。
- `outputs/`：两段最终吴语回答和机器评估报告。
- `tests/`：自动化测试。

## 大文件说明

GitHub 不存放虚拟环境、公开数据音频和模型权重。它们体积约数 GB，而且均可从官方来源恢复：

```powershell
SETUP.bat
powershell -ExecutionPolicy Bypass -File scripts\setup_whisper_medium_wu.ps1
powershell -ExecutionPolicy Bypass -File scripts\setup_wenet_wu_sft.ps1
```

公开上海话数据可运行：

```powershell
.venv\Scripts\python.exe scripts\fetch_shanghai_hf_dataset.py
```

## 运行

```text
双击 START_UI.bat
```

浏览器访问 `http://localhost:8501`。

实时对话：

```text
双击 START_LIVE_AGENT.bat
```

或运行：

```powershell
.venv\Scripts\python.exe -m ganagent.live_agent --reply-target wuu
```

## 已知限制

- 第一次安装需要下载数 GB 模型。
- 当前公开 Wu-SFT 偶尔会出现句首赘词，因此系统使用回听仲裁，而不是直接发布第一份生成结果。
- 两个样例通过不等于商用认证；商用前仍需更大规模母语者听测。
