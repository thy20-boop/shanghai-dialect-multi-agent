from __future__ import annotations

from ganagent.asr_backends import ASRBackend
from ganagent.dialect import ShanghaiDialectDetector
from ganagent.models import AgentResult, Segment, Suspicion
from ganagent.repair import RepairEngine, count_repair_actions


class ShanghaiDialectAgent:
    def __init__(
        self,
        asr_backend: ASRBackend,
        detector: ShanghaiDialectDetector | None = None,
        repair_engine: RepairEngine | None = None,
    ) -> None:
        self.asr_backend = asr_backend
        self.detector = detector or ShanghaiDialectDetector()
        self.repair_engine = repair_engine or RepairEngine()

    def run(self, audio_path: str | None = None) -> AgentResult:
        agent_trace: list[dict[str, object]] = [
            {
                "agent": "音频预处理智能体",
                "role": "输入规范化",
                "status": "ok",
                "summary": "接收音频/视频文件，并交给 ASR 后端统一转为 16k 单声道音频特征。",
            }
        ]
        segments = self.asr_backend.transcribe(audio_path)
        alternatives = list(getattr(self.asr_backend, "alternatives", []) or [])
        agent_trace.extend(list(getattr(self.asr_backend, "agent_trace", []) or []))
        raw_transcript = "".join(segment.text for segment in segments)
        dialect = self.detector.detect(raw_transcript)
        repairs = self.repair_engine.repair_segments(segments)
        repairs.extend(self.repair_engine.repair_segments_with_alternatives(segments, alternatives))
        repairs.extend(self.repair_engine.rerank_segments_with_alternatives(segments, alternatives))
        agent_trace.append(
            {
                "agent": "候选仲裁智能体",
                "role": "多候选重排",
                "status": "candidate_adopted"
                if any(item.get("type") == "alternative_rerank" for item in repairs)
                else "primary_kept",
                "summary": "比较 Dolphin 主输出与本地 LoRA 候选，仅在主输出明显高风险时才切换候选。",
                "candidate_count": sum(1 for item in alternatives if item.get("status") == "ok"),
            }
        )
        suspicions = self.repair_engine.find_suspicions(segments, repairs)
        transcript = "".join(segment.display_text() for segment in segments)
        mandarin_translation, translation_repairs = (
            self.repair_engine.translate_to_mandarin_with_replacements(transcript)
        )
        repairs = repairs + translation_repairs
        agent_trace.extend(
            [
                {
                    "agent": "纠错记忆智能体",
                    "role": "词典/记忆修复",
                    "status": "ok",
                    "summary": "应用上海话词典、上下文规则和用户长期纠错记忆。",
                    "repair_count": count_repair_actions(repairs),
                },
                {
                    "agent": "上海话转普通话智能体",
                    "role": "语义转换",
                    "status": "ok",
                    "summary": "把上海话词汇和口语表达转换为普通话文本。",
                },
                {
                    "agent": "风险检测智能体",
                    "role": "质量评估",
                    "status": "high_risk" if any(item.severity == "high" for item in suspicions) else "ok",
                    "summary": "检测重复幻觉、乱码、低置信度、方言残留和需要人工复核的片段。",
                    "suspicion_count": len(suspicions),
                },
            ]
        )
        active_learning_items = self._build_active_learning_items(
            audio_path,
            segments,
            alternatives,
            suspicions,
            repairs,
        )
        agent_trace.append(
            {
                "agent": "主动学习智能体",
                "role": "再训练样本挖掘",
                "status": "needs_review" if active_learning_items else "no_action",
                "summary": "收集高风险片段、候选分歧和人工可纠错内容，后续可进入再训练集。",
                "item_count": len(active_learning_items),
            }
        )

        return AgentResult(
            audio_path=audio_path,
            dialect=dialect,
            segments=segments,
            suspicions=suspicions,
            transcript=transcript,
            mandarin_translation=mandarin_translation,
            repairs=repairs,
            alternatives=alternatives,
            agent_trace=agent_trace,
            active_learning_items=active_learning_items,
        )

    @staticmethod
    def _build_active_learning_items(
        audio_path: str | None,
        segments: list[Segment],
        alternatives: list[dict],
        suspicions: list[Suspicion],
        repairs: list[dict],
    ) -> list[dict[str, object]]:
        transcript = "".join(segment.display_text() for segment in segments)
        candidate_texts = [
            str(item.get("transcript") or "").strip()
            for item in alternatives
            if item.get("status") == "ok" and str(item.get("transcript") or "").strip()
        ]
        reasons: set[str] = set()
        if any(item.severity == "high" for item in suspicions):
            reasons.add("high_risk_suspicion")
        if candidate_texts and any(candidate != transcript for candidate in candidate_texts):
            reasons.add("asr_candidate_disagreement")
        if any(item.get("type") == "alternative_rerank" for item in repairs):
            reasons.add("candidate_rerank_adopted")
        if any(item.get("type") == "asr_repair" for item in repairs):
            reasons.add("post_asr_repair")

        if not reasons:
            return []
        return [
            {
                "audio_path": audio_path,
                "reason": sorted(reasons),
                "primary_transcript": transcript,
                "candidate_transcripts": candidate_texts,
                "suggested_action": "人工确认正确文本后写入 data/user_corrections.json，或加入下一轮 LoRA 微调清单。",
            }
        ]


# Backward-compatible import for scripts built against the original package.
GanDialectAgent = ShanghaiDialectAgent
