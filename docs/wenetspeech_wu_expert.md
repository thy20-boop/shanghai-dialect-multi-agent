# WenetSpeech-Wu 吴语生成专家

## 当前架构

系统仍然是多智能体，而不是单独调用一个 TTS 模型：

1. Dolphin 负责上海话主识别。
2. 本地 Whisper-LoRA 提供第二候选。
3. 候选仲裁、纠错记忆和风险检测产生可靠的普通话结果。
4. 上海话表达模块把回答改写成吴语口语稿。
5. WenetSpeech-Wu `CosyVoice2-Wu-SFT` 按句生成确定性候选。
6. Whisper-Medium-Wu 回识候选音频。
7. 语音质量智能体检查关键词召回、方言信号和可疑片段。
8. 仲裁智能体输出最终 MP3。

## 本机部署

- 独立 Python 3.10 环境：`%SHANGHAI_WU_RUNTIME%\.venv`
- 官方 CosyVoice 源码：`%SHANGHAI_WU_RUNTIME%\CosyVoice`
- 吴语专家模型：`%SHANGHAI_WU_RUNTIME%\models\CosyVoice2-Wu-SFT-runtime`
- 上海话参考音频：`assets\reference\official_shanghai_prompt.wav`
- 本地服务：`http://127.0.0.1:9881/tts`
- 启动脚本：`scripts\start_wenet_wu_expert.ps1`

安装命令：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_wenet_wu_sft.ps1
```

## 验证结果

在原来的两个求助回答上：

- 身份证回答整段字符准确率为 `85.87%`，关键实体召回为 `100%`。
- 求助电话回答整段字符准确率为 `76.64%`，五个电话号码全部保留。
- 66 项自动化测试通过。

模型解决了旧方案使用普通话音素前端朗读上海话文字的问题。后续提升重点应放在上海话回答文本规范化、数字读法和更强的吴语回识模型，而不是继续训练旧 GPT-SoVITS。
