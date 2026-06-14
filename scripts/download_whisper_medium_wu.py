import os
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from huggingface_hub import hf_hub_download


REPO_ID = "ASLP-lab/WenetSpeech-Wu-Speech-Understanding"
legacy = Path(r"D:\wswu_runtime")
runtime_root = Path(
    os.environ.get("SHANGHAI_WU_RUNTIME")
    or (legacy if legacy.exists() else Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "ShanghaiDialectAgent" / "wswu_runtime")
)
LOCAL_DIR = str(runtime_root / "models" / "Whisper-Medium-Wu")


for filename in ("whisper/train.yaml", "whisper/whisper-medium.pt"):
    print(hf_hub_download(REPO_ID, filename, local_dir=LOCAL_DIR))
