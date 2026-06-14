Shanghai Dialect ASR Online Package

This small package does NOT include Python, PyTorch, or the Whisper model.
It requires:
- Windows 10/11 64-bit
- Python 3.10-3.12 available as "python" in PATH
- Internet access on first run
- Several GB of free disk space for dependencies and the model cache

How to use:
1. Unzip the package.
2. Double-click START_UI.bat.
3. The first run installs dependencies and may take several minutes.
4. Upload a WAV/FLAC/OGG/M4A/MP3/MP4/AAC file in the web UI.

Long MP4/audio files are segmented from the audio track by pauses first. This
does not use video subtitles or screen text.

Other entry points:
- SETUP.bat: install dependencies only.
- RUN_SAMPLE.bat: run the included Shanghai sample.
- TRANSCRIBE_AUDIO.bat: drag an audio file onto it for JSON output.

The GitHub submission does not duplicate multi-GB model weights. Official ASR and
Wu TTS models download on first use and are reused from the local cache afterwards.

Experimental open-source ASR candidates are exposed in the advanced settings for
comparison. Install them separately with:
  python -m pip install -r requirements-experimental-asr.txt

The default trained Whisper-LoRA model remains the supported handoff path.
