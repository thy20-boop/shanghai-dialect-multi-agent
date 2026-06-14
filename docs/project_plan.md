# 上海话 ASR 修复 Agent 项目方案

## 目标

输入上海话音频/视频，输出普通话转换、原始转写、方言信号、局部修复记录、候选分歧、待复核片段和主动学习样本。项目核心从单一 ASR 模型升级为“Dolphin 强识别底座 + 自研多智能体协作层”。

## 流程

```text
上海话音频
  -> 音频预处理智能体
  -> Dolphin ASR 专家主识别
  -> 本地 Whisper-LoRA 复核/兜底
  -> 候选仲裁智能体
  -> 纠错记忆智能体
  -> 普通话转换
  -> 风险检测智能体
  -> 主动学习队列
  -> 人工确认后导出下一轮训练 manifest
  -> 报告与人工复核
```

## 数据与模型

- 数据：`TingChen-ppmc/Shanghai_Dialect_Conversational_Speech_Corpus`
- 主识别底座：`Dolphin small.cn`
- 上海话 baseline：`TingChen-ppmc/whisper-small-Shanghai`
- 自训练：3700 条 Whisper LoRA，作为复核、兜底和下一轮学习基线
- 纠错记忆：`data/user_corrections.json`
- 主动学习：`data/active_learning_queue.jsonl`
- 质量评分：综合高风险提示、候选共识、修复行为和主动学习状态

## 指标

- CER
- 上海话线索词召回率
- 技术术语召回率
- 完全匹配率
- 高风险片段拦截率
- 候选分歧发现数
- 主动学习样本有效率
- 质量评分分布
- 已确认主动学习样本导出数量

## 风险

- 上海话汉字写法并不完全统一，字符级指标不能覆盖全部语言质量。
- 通用 Whisper 可能把上海话普通话化。
- Dolphin 虽然更强，但仍可能在长视频、背景声、人名地名和口语快读上出错。
- 规则翻译适合演示和可解释修复，不等同于完整机器翻译。
- 主动学习样本需要人工确认后才能进入下一轮训练，不能自动把不确定输出当标签。
