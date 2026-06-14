from __future__ import annotations

import re
from dataclasses import dataclass

from ganagent.evaluation import levenshtein_distance, normalize_text
from ganagent.models import AgentResult
from ganagent.repair import count_repair_actions


REPETITION_RE = re.compile(r"(.{2,12}?)(?:\1){3,}")
MAX_COMPACTED_CHARS = 220


@dataclass
class TranslationProduct:
    mandarin: str
    dialect_transcript: str
    status: str
    status_label: str
    warning: str | None
    suspicion_count: int
    repair_count: int
    agent_count: int = 0
    active_learning_count: int = 0
    quality_score: float = 1.0
    consensus_score: float | None = None
    action_suggestion: str | None = None
    text_compacted: bool = False
    draft_mandarin: str | None = None
    draft_dialect_transcript: str | None = None
    failure_reasons: list[str] | None = None

    def as_dict(self) -> dict[str, str | int | float | bool | list[str] | None]:
        return {
            "mandarin": self.mandarin,
            "dialect_transcript": self.dialect_transcript,
            "status": self.status,
            "status_label": self.status_label,
            "warning": self.warning,
            "suspicion_count": self.suspicion_count,
            "repair_count": self.repair_count,
            "agent_count": self.agent_count,
            "active_learning_count": self.active_learning_count,
            "quality_score": self.quality_score,
            "consensus_score": self.consensus_score,
            "action_suggestion": self.action_suggestion,
            "text_compacted": self.text_compacted,
            "draft_mandarin": self.draft_mandarin,
            "draft_dialect_transcript": self.draft_dialect_transcript,
            "failure_reasons": self.failure_reasons,
        }


def build_translation_product(result: AgentResult) -> TranslationProduct:
    mandarin, mandarin_compacted = _compact_repeated_text(result.mandarin_translation)
    dialect_transcript, transcript_compacted = _compact_repeated_text(result.transcript)
    contains_repetition_marker = "[重复片段省略]" in mandarin or "[重复片段省略]" in dialect_transcript
    high_risk = [item for item in result.suspicions if item.severity == "high"]
    warning = None
    status = "ok"
    quality_score = estimate_quality_score(result)
    consensus_score = estimate_consensus_score(result)
    if high_risk:
        status = "unreliable"
        status_label = "无法可靠识别"
        failure_reasons = sorted({item.reason for item in high_risk})
        warning = "识别结果包含高风险可疑片段，主输出已拦截；请查看草稿或换更短、更清晰片段重跑。"
        final_mandarin = "无法可靠识别。建议切成 3-5 秒短音频、降低背景声，或调小长音频最大切片秒数后重跑。"
        final_transcript = "无法可靠识别。"
    elif result.suspicions:
        status = "usable_with_notes"
        status_label = "可用但需留意"
        warning = "识别结果包含低/中风险提示。"
        failure_reasons = []
        final_mandarin = mandarin
        final_transcript = dialect_transcript
    else:
        status_label = "可直接使用"
        failure_reasons = []
        final_mandarin = mandarin
        final_transcript = dialect_transcript

    action_suggestion = suggest_next_action(status, result.active_learning_items, result.suspicions)
    return TranslationProduct(
        mandarin=final_mandarin,
        dialect_transcript=final_transcript,
        status=status,
        status_label=status_label,
        warning=warning,
        suspicion_count=len(result.suspicions),
        repair_count=count_repair_actions(result.repairs),
        agent_count=len(result.agent_trace),
        active_learning_count=len(result.active_learning_items),
        quality_score=quality_score,
        consensus_score=consensus_score,
        action_suggestion=action_suggestion,
        text_compacted=mandarin_compacted or transcript_compacted or contains_repetition_marker,
        draft_mandarin=mandarin if high_risk else None,
        draft_dialect_transcript=dialect_transcript if high_risk else None,
        failure_reasons=failure_reasons,
    )


def estimate_quality_score(result: AgentResult) -> float:
    severity_penalty = {"high": 0.55, "medium": 0.12, "low": 0.025}
    penalty = 0.0
    for suspicion in result.suspicions:
        penalty += severity_penalty.get(suspicion.severity, 0.08)
    if result.active_learning_items:
        penalty += min(0.10, len(result.active_learning_items) * 0.05)
    if any(item.get("type") == "alternative_rerank" for item in result.repairs):
        penalty += 0.06
    score = max(0.0, min(1.0, 1.0 - penalty))
    return round(score, 3)


def estimate_consensus_score(result: AgentResult) -> float | None:
    candidates = [
        str(item.get("transcript") or "").strip()
        for item in result.alternatives
        if item.get("status") == "ok" and str(item.get("transcript") or "").strip()
    ]
    if not candidates:
        return None
    primary = normalize_text(result.transcript)
    if not primary:
        return 0.0
    scores: list[float] = []
    for candidate in candidates:
        normalized_candidate = normalize_text(candidate)
        if not normalized_candidate:
            scores.append(0.0)
            continue
        distance = levenshtein_distance(primary, normalized_candidate)
        denominator = max(len(primary), len(normalized_candidate), 1)
        scores.append(max(0.0, 1.0 - distance / denominator))
    return round(sum(scores) / len(scores), 3)


def suggest_next_action(
    status: str,
    active_learning_items: list[dict],
    suspicions: list,
) -> str:
    if status == "unreliable":
        return "建议人工复核或重新切分音频；不要直接把当前结果作为最终转写。"
    if active_learning_items:
        return "建议把该样本保留在主动学习队列，人工确认后用于下一轮纠错记忆或 LoRA 微调。"
    if suspicions:
        return "结果可用，但建议检查提示片段，特别是人名、地名和方言词。"
    return "结果可直接使用，也可作为高质量样本加入展示集。"


def _compact_repeated_text(text: str) -> tuple[str, bool]:
    compacted = False
    cleaned = text.replace("\ufffd", "")
    if cleaned != text:
        compacted = True

    for _ in range(4):
        updated = REPETITION_RE.sub(lambda match: f"{match.group(1)}{match.group(1)}[重复片段省略]", cleaned)
        if updated == cleaned:
            break
        compacted = True
        cleaned = updated

    cleaned = re.sub(r"(?:\[重复片段省略\]){2,}", "[重复片段省略]", cleaned)
    cleaned = cleaned.strip(" ，,")
    if compacted and len(cleaned) > MAX_COMPACTED_CHARS:
        cleaned = cleaned[:MAX_COMPACTED_CHARS].strip(" ，,") + "[长重复片段省略]"
    return cleaned, compacted
