# Colleague Quickstart

这是上海话转普通话 agent 的轻量交付版。

## 安装

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_colleague.ps1
```

## Web UI

```powershell
.venv\Scripts\python.exe -m streamlit run app\streamlit_app.py --server.port 8501
```

打开 `http://localhost:8501`，上传上海话 WAV、FLAC 或 OGG。

GitHub 提交版不包含数 GB 的公开模型缓存。首次启动 `START_UI.bat` 会按需下载 Whisper-Medium-Wu 和 CosyVoice2-Wu-SFT。吴语生成环境也可以单独安装：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_wenet_wu_sft.ps1
```

如果要尝试其他开源候选后端，例如 FunASR/SenseVoice 或 Dolphin，可以额外安装：

```powershell
python -m pip install -r requirements-experimental-asr.txt
```

然后在网页高级设置里切换识别后端和模型预设。

## CLI

```powershell
.\scripts\translate_audio.ps1 -Audio path\to\shanghai.wav -Json -Online
```

模型下载完成并缓存后可以去掉 `-Online`。

## Mock 验证

```powershell
.\scripts\run_all_mock.ps1
```

或：

```powershell
python scripts\run_tests.py
```

包内没有把旧南昌话音频冒充上海话示例。真实 ASR 需要自行上传上海话音频，或先运行 `scripts/fetch_shanghai_hf_dataset.py` 获取公开语料。
