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

## 可复现结果

| 测试回答 | 整段字符准确率 | 关键实体召回 |
| --- | ---: | ---: |
| 身份证求助 | 85.87% | 100% |
| 紧急求助电话 | 76.64% | 100% |

测试命令：

```powershell
.venv\Scripts\python.exe -m pytest -q
```

当前版本共有 66 项自动化测试。

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

## 已知限制

- 第一次安装需要下载数 GB 模型。
- 当前公开 Wu-SFT 偶尔会出现句首赘词，因此系统使用回听仲裁，而不是直接发布第一份生成结果。
- 两个样例通过不等于商用认证；商用前仍需更大规模母语者听测。
