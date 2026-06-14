from __future__ import annotations

from pathlib import Path
from typing import Any

from ganagent.models import AgentResult
from ganagent.product import TranslationProduct


def render_codex_answer_task(
    product: TranslationProduct,
    result: AgentResult,
    audio_path: str | Path | None = None,
) -> str:
    """Render a Codex-native web-answer task from an ASR result.

    The generated file is intentionally a handoff prompt, not a web app feature.
    Codex reads it, searches the web with its own tools, and writes the answer.
    """

    suspicions = "\n".join(
        f"- {item.severity}: {item.reason}；{item.evidence}；建议：{item.suggestion}"
        for item in result.suspicions
    ) or "- 无"
    agents = "\n".join(
        f"- {item.get('agent', 'unknown')}：{item.get('summary', '')}"
        for item in result.agent_trace
    ) or "- 无"
    alternatives = "\n".join(
        f"- {item.get('backend', item.get('agent', 'candidate'))}: {item.get('transcript', item.get('status', ''))}"
        for item in result.alternatives
    ) or "- 无"
    active_items = "\n".join(
        f"- {', '.join(map(str, item.get('reason', [])))}：{item.get('suggested_action', '')}"
        for item in result.active_learning_items
    ) or "- 无"
    audio = str(audio_path) if audio_path else "未提供"

    return f"""# Codex 联网问答任务

## 用户问题候选
{product.mandarin}

## 识别原文
{product.dialect_transcript}

## 音频文件
{audio}

## 识别质量
- 状态：{product.status_label} ({product.status})
- 质量评分：{product.quality_score:.3f}
- 修复次数：{product.repair_count}
- 可疑片段数：{product.suspicion_count}
- 候选共识：{product.consensus_score if product.consensus_score is not None else "无"}
- 建议：{product.action_suggestion or "无"}

## 可疑片段
{suspicions}

## 协作智能体记录
{agents}

## 其他识别候选
{alternatives}

## 主动学习候选
{active_items}

## Codex 执行要求
1. 先判断“用户问题候选”是否真的是一个可回答的问题；如果识别质量低或问题不完整，先说明不确定点。
2. 如涉及新闻、地点、政策、价格、人物、产品、课程资料等可能变化的信息，必须由 Codex 联网搜索后再回答。
3. 优先使用官方来源、原始资料或可信新闻来源；回答中保留来源链接。
4. 回答要用中文，尽量短而清楚；如果检索证据不足，要明确说“目前没有足够证据”。
5. 把最终回答写入 `outputs/codex_answer.txt`，再同时生成普通话 MP3、吴语口语稿文本和吴语口语稿 MP3：

```powershell
.\\.venv\\Scripts\\python.exe -m ganagent.cli speak --text-file outputs\\codex_answer.txt --output outputs\\codex_answer.mp3 --wu-output outputs\\codex_answer_wu.mp3 --wu-text-output outputs\\codex_answer_wu.txt
```

6. 最终交付里同时列出普通话 MP3、吴语文本和吴语 MP3 路径。若当前 TTS 后端没有真实 wuu-CN/上海话音色，必须说明吴语 MP3 只是“训练集风格吴语口语稿 + 普通话音色朗读”。
"""


def render_codex_task_metadata(product: TranslationProduct, output_path: str | Path) -> dict[str, Any]:
    return {
        "codex_task_output": str(output_path),
        "question_candidate": product.mandarin,
        "status": product.status,
        "quality_score": product.quality_score,
        "suspicion_count": product.suspicion_count,
    }
