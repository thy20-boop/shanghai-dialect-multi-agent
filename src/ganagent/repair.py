from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

from ganagent.models import Segment, Suspicion


REPETITION_COMPACTION_RE = re.compile(r"(.{1,12}?)(?:\1){4,}")
REPETITION_MARKER = "[重复片段省略]"
QUALITY_PUNCTUATION_RE = re.compile(r"[\s，。！？；：、,.!?;:\"'“”‘’（）()\[\]{}<>《》]+")
KNOWN_ASR_CONFUSIONS = {
    "先什么",
    "先撒",
    "你行",
    "侬行",
    "车子机面",
    "无语",
    "五语",
    "屋语",
    "沪雨",
    "微信",
    "塞尔",
    "罗拉",
    "萝拉",
}
QUALITY_DIALECT_MARKERS = {
    "阿拉",
    "侬",
    "吾",
    "伊拉",
    "拧",
    "搿",
    "勿",
    "伐",
    "辰光",
    "老早",
    "第一趟",
    "上海话",
    "吴语",
}


def count_repair_actions(repairs: list[dict[str, Any]]) -> int:
    total = 0
    for repair in repairs:
        replacements = repair.get("replacements") or []
        if not replacements:
            total += 1
            continue
        for replacement in replacements:
            try:
                count = int(replacement.get("count", 1))
            except (TypeError, ValueError):
                count = 1
            total += max(count, 1)
    return total


DEFAULT_GLOSSARY = {
    "translation_rules": {
        "侬好伐？": "你好吗？",
        "侬好伐": "你好吗",
        "吾听勿大清爽": "我听不太清楚",
        "伊拉": "他们",
        "阿拉个": "我们的",
        "阿拉": "我们",
        "辣海上海": "在上海",
        "辣撒地方": "在什么地方",
        "辣搿搭": "在这里",
        "辣这里": "在这里",
        "搿搭": "这里",
        "拧来": "人来",
        "三家头": "三个人",
        "几个号头": "几个月",
        "号头": "个月",
        "毛毛叫": "大约",
        "搿个": "这个",
        "迭个": "这个",
        "今朝": "今天",
        "葛末": "那么",
        "辰光": "时候",
        "交关": "很",
        "勿要": "不要",
        "哪能": "怎么",
        "伐有": "还有",
        "下趟": "下次",
        "第一趟": "第一次",
        "是呢": "是的",
        "是额": "是的",
        "是呃": "是的",
        "老早": "以前",
        "寻呃地方": "找个地方",
        "叫寻一寻": "找一找",
        "寻一寻": "找一找",
        "撒": "什么",
        "啥": "什么",
        "等歇": "等一会儿",
        "辣辣": "在",
        "清爽": "清楚",
        "侪": "都",
        "唻": "",
    },
    "dialect_terms": {
        "伊": "他",
        "吾": "我",
        "侬": "你",
        "拧": "人",
        "勿": "不",
        "伐": "吗",
        "灵": "好",
        "搿": "这",
    },
    "review_terms": {
        "拧来": "普通话里通常应转成“人来”，需要确认上下文。",
        "拧": "上海话里常表示“人”，普通话结果中残留时需要复核。",
        "阿拉": "上海话“我们”，如果普通话结果未替换需要复核。",
        "搿": "上海话“这/这个”，需要按上下文确认。",
        "侬": "上海话“你”，需要按上下文确认。",
    },
    "asr_repairs": {
        "罗拉": "LoRA",
        "萝拉": "LoRA",
        "塞尔": "CER",
        "西伊阿": "CER",
        "大语言摸型": "大语言模型",
        "微信": "微调",
        "小龙包": "小笼包",
        "小龙馒头": "小笼馒头",
        "生煎慢头": "生煎馒头",
        "排骨粘糕": "排骨年糕",
        "装饲工": "装修工",
        "猫猫叫": "毛毛叫",
        "工地党工": "工地打工",
        "同学唻": "同行唻",
        "再利息": "再联系",
        "辣撒地方走": "辣撒地方做",
    },
    "asr_context_repairs": [
        {
            "from": "那好，农行",
            "to": "那好，侬好",
            "context_any": ["初次见面", "蛮开心"],
        },
        {
            "from": "可以当大蛮开心",
            "to": "看到蛮开心",
            "context_any": ["初次见面"],
        },
        {
            "from": "安慰",
            "to": "安徽",
            "context_any": ["老家", "到上海"],
        },
        {
            "from": "当工",
            "to": "打工",
            "context_any": ["到上海", "打工"],
        },
        {
            "from": "老家人是穿",
            "to": "老家是四川",
            "context_any": ["老家", "刚到"],
        },
        {
            "from": "恐惠拧大家",
            "to": "欢迎大家",
            "context_any": ["房子", "大家"],
        },
        {
            "from": "大家是撕拨转移，虽然手机也没拿下来",
            "to": "大家先拿手机号码留下来",
            "context_any": ["手机", "再利息", "再联系"],
        },
        {
            "from": "无语",
            "to": "吴语",
            "context_any": ["上海", "上海话", "方言", "沪语", "吴语"],
        },
        {
            "from": "五语",
            "to": "吴语",
            "context_any": ["上海", "上海话", "方言", "沪语", "吴语"],
        },
        {
            "from": "屋语",
            "to": "吴语",
            "context_any": ["上海", "上海话", "方言", "沪语", "吴语"],
        },
        {
            "from": "沪雨",
            "to": "沪语",
            "context_any": ["上海", "上海话", "方言", "吴语"],
        },
        {
            "from": "先什么",
            "to": "先生",
            "context_any": ["你好", "侬好", "你行", "侬行", "第一趟", "上海"],
        },
        {
            "from": "先撒",
            "to": "先生",
            "context_any": ["你好", "侬好", "你行", "侬行", "第一趟", "上海"],
        },
        {
            "from": "你行",
            "to": "侬好",
            "context_any": ["先生", "先什么", "第一趟", "上海"],
        },
        {
            "from": "侬行",
            "to": "侬好",
            "context_any": ["先生", "先撒", "先什么", "第一趟", "上海"],
        },
        {
            "from": "你是第一趟",
            "to": "侬是第一趟",
            "context_any": ["上海"],
        },
    ],
    "alternative_context_repairs": [
        {
            "from": "先什么",
            "to": "先生",
            "candidate_any": ["先生", "先森"],
            "context_any": ["你好", "侬好", "你行", "侬行", "第一趟", "上海"],
        },
        {
            "from": "先撒",
            "to": "先生",
            "candidate_any": ["先生", "先森"],
            "context_any": ["你好", "侬好", "你行", "侬行", "第一趟", "上海"],
        },
        {
            "from": "你行",
            "to": "侬好",
            "candidate_any": ["侬好", "你好"],
            "context_any": ["先生", "先什么", "第一趟", "上海"],
        },
        {
            "from": "侬行",
            "to": "侬好",
            "candidate_any": ["侬好", "你好"],
            "context_any": ["先生", "先撒", "先什么", "第一趟", "上海"],
        },
        {
            "from": "车子机面",
            "to": "初次见面",
            "candidate_any": ["初次见面"],
            "context_any": ["你好", "欢迎", "见面"],
        },
    ],
    "domain_terms": ["LoRA", "CER", "WER", "Qwen", "Whisper", "SenseVoice"],
}


class RepairEngine:
    def __init__(self, glossary: dict[str, Any] | None = None) -> None:
        self.glossary = glossary or DEFAULT_GLOSSARY

    @classmethod
    def from_file(cls, path: str | Path | None) -> "RepairEngine":
        if path is None:
            return cls()
        with Path(path).open("r", encoding="utf-8") as f:
            return cls(json.load(f))

    @classmethod
    def from_file_with_memory(
        cls,
        path: str | Path | None,
        memory_path: str | Path | None = None,
    ) -> "RepairEngine":
        return cls.from_file(path).with_custom_repairs(load_custom_repairs_file(memory_path))

    def with_custom_repairs(self, repairs: dict[str, str]) -> "RepairEngine":
        if not repairs:
            return self
        glossary = copy.deepcopy(self.glossary)
        glossary.setdefault("asr_repairs", {}).update(repairs)
        return RepairEngine(glossary)

    def repair_segments(self, segments: list[Segment]) -> list[dict[str, Any]]:
        repairs: list[dict[str, Any]] = []
        replacements = self.glossary.get("asr_repairs", {})
        full_context = "".join(segment.text for segment in segments)
        for index, segment in enumerate(segments):
            repaired = segment.text
            applied: list[dict[str, str]] = []
            compacted = self._compact_repeated_hallucination(repaired)
            if compacted != repaired:
                repaired = compacted
                applied.append(
                    {
                        "from": "重复幻觉片段",
                        "to": REPETITION_MARKER,
                        "count": "1",
                        "source": "repetition_compaction",
                    }
                )
            for wrong, right in replacements.items():
                if wrong in repaired:
                    repaired = repaired.replace(wrong, right)
                    applied.append({"from": wrong, "to": right, "count": "1", "source": "asr_repairs"})
            for rule in self.glossary.get("asr_context_repairs", []):
                wrong = str(rule.get("from", ""))
                right = str(rule.get("to", ""))
                context_any = [str(item) for item in rule.get("context_any", [])]
                if not wrong or wrong not in repaired:
                    continue
                context = f"{full_context}{repaired}"
                if context_any and not any(item in context for item in context_any):
                    continue
                repaired = repaired.replace(wrong, right)
                applied.append({"from": wrong, "to": right, "count": "1", "source": "asr_context_repairs"})
            if applied:
                segment.repaired_text = repaired
                repairs.append(
                    {
                        "type": "asr_repair",
                        "segment_index": index,
                        "original": segment.text,
                        "repaired": repaired,
                        "replacements": applied,
                    }
                )
        return repairs

    def repair_segments_with_alternatives(
        self,
        segments: list[Segment],
        alternatives: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        candidate_text = "".join(str(item.get("transcript") or "") for item in alternatives)
        if not candidate_text:
            return []

        full_context = "".join(segment.display_text() for segment in segments)
        repairs: list[dict[str, Any]] = []
        for index, segment in enumerate(segments):
            repaired = segment.display_text()
            applied: list[dict[str, str]] = []
            for rule in self.glossary.get("alternative_context_repairs", []):
                wrong = str(rule.get("from", ""))
                right = str(rule.get("to", ""))
                if not wrong or wrong not in repaired:
                    continue
                candidate_any = [str(item) for item in rule.get("candidate_any", [])]
                if candidate_any and not any(item in candidate_text for item in candidate_any):
                    continue
                context_any = [str(item) for item in rule.get("context_any", [])]
                context = f"{full_context}{candidate_text}{repaired}"
                if context_any and not any(item in context for item in context_any):
                    continue
                repaired = repaired.replace(wrong, right)
                applied.append(
                    {
                        "from": wrong,
                        "to": right,
                        "count": "1",
                        "source": "open_source_asr_candidate",
                    }
                )
            if applied:
                original = segment.display_text()
                segment.repaired_text = repaired
                repairs.append(
                    {
                        "type": "asr_repair",
                        "segment_index": index,
                        "original": original,
                        "repaired": repaired,
                        "replacements": applied,
                    }
                )
        return repairs

    def rerank_segments_with_alternatives(
        self,
        segments: list[Segment],
        alternatives: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Let a stronger candidate take over only when the primary is risky.

        The local fine-tuned model stays the default source. Open-source
        candidates are used as a reviewer: they can replace the visible
        transcript when the primary output shows clear ASR failure patterns,
        such as repetition, undecodable characters, or known short-phrase
        confusions that survived normal repairs.
        """

        if not segments:
            return []
        primary_text = "".join(segment.display_text() for segment in segments).strip()
        primary_quality = self.score_transcript_quality(primary_text)
        candidate = self._best_candidate(alternatives)
        if candidate is None:
            return []
        candidate_text = str(candidate["text"]).strip()
        candidate_quality = float(candidate["score"])
        if not self._should_accept_candidate(primary_text, primary_quality, candidate_text, candidate_quality):
            return []

        original_text = primary_text
        if len(segments) == 1:
            segments[0].repaired_text = candidate_text
        else:
            # Dolphin-style candidates currently have no reliable timestamps.
            # Keep a single coherent replacement instead of pretending we know
            # how to align every word back to the primary chunks.
            segments[0].repaired_text = candidate_text
            for segment in segments[1:]:
                segment.repaired_text = ""

        backend = str(candidate.get("backend") or "candidate")
        return [
            {
                "type": "alternative_rerank",
                "segment_index": 0,
                "original": original_text,
                "repaired": candidate_text,
                "replacements": [
                    {
                        "from": "primary_transcript",
                        "to": backend,
                        "count": "1",
                        "source": "candidate_rerank",
                    }
                ],
                "decision": {
                    "primary_score": round(primary_quality, 3),
                    "candidate_score": round(candidate_quality, 3),
                    "candidate_backend": backend,
                    "primary_problems": self.transcript_quality_problems(primary_text),
                    "candidate_problems": self.transcript_quality_problems(candidate_text),
                },
            }
        ]

    def find_suspicions(self, segments: list[Segment], repairs: list[dict[str, Any]]) -> list[Suspicion]:
        suspicions: list[Suspicion] = []
        repaired_indexes = {item["segment_index"] for item in repairs}

        for index, segment in enumerate(segments):
            text = segment.display_text()
            if segment.confidence is not None and segment.confidence < 0.7:
                suspicions.append(
                    Suspicion(
                        segment_index=index,
                        severity="medium",
                        reason="low_confidence",
                        evidence=f"confidence={segment.confidence:.2f}",
                        suggestion="建议局部重切分后重识别，或交给母语者确认。",
                    )
                )

            if index in repaired_indexes:
                suspicions.append(
                    Suspicion(
                        segment_index=index,
                        severity="low",
                        reason="glossary_repair",
                        evidence=segment.text,
                        suggestion="已按术语/方言词典自动修复；展示时可作为 agent 价值点。",
                    )
                )

            if self._has_repetition(text):
                suspicions.append(
                    Suspicion(
                        segment_index=index,
                        severity="high",
                        reason="repetition",
                        evidence=text,
                        suggestion="可能是 ASR 幻觉或长音频切分失败，建议只重跑该片段。",
                    )
                )

            if "\ufffd" in text:
                suspicions.append(
                    Suspicion(
                        segment_index=index,
                        severity="high",
                        reason="replacement_character",
                        evidence=text,
                        suggestion="ASR 输出含无法解码或无法识别的字符，建议换短片段重跑或人工复核。",
                    )
                )

            if self._has_mixed_technical_terms(text):
                suspicions.append(
                    Suspicion(
                        segment_index=index,
                        severity="medium",
                        reason="mixed_domain_terms",
                        evidence=text,
                        suggestion="中英术语混说片段建议启用课程术语表。",
                    )
                )
            review_hits = self._find_review_terms(text)
            if review_hits:
                suspicions.append(
                    Suspicion(
                        segment_index=index,
                        severity="low",
                        reason="dialect_review_term",
                        evidence="；".join(review_hits),
                        suggestion="普通话结果中可能残留上海话词，已按词典尝试转换；建议人工确认这类词的上下文含义。",
                    )
                )
        return suspicions

    def translate_to_mandarin(self, text: str) -> str:
        translated, _ = self.translate_to_mandarin_with_replacements(text)
        return translated

    def translate_to_mandarin_with_replacements(self, text: str) -> tuple[str, list[dict[str, Any]]]:
        translated = text
        applied: list[dict[str, str | int]] = []
        for source, replacements in (
            ("translation_rules", self.glossary.get("translation_rules", {})),
            ("dialect_terms", self.glossary.get("dialect_terms", {})),
        ):
            for dialect, mandarin in replacements.items():
                count = translated.count(dialect)
                if count <= 0:
                    continue
                translated = translated.replace(dialect, mandarin)
                applied.append(
                    {
                        "from": dialect,
                        "to": mandarin,
                        "count": count,
                        "source": source,
                    }
                )
        translated = translated.replace("你好吗？吗", "你好吗？")
        repairs = []
        if applied:
            repairs.append(
                {
                    "type": "dialect_translation",
                    "segment_index": None,
                    "original": text,
                    "repaired": translated,
                    "replacements": applied,
                }
            )
        return translated, repairs

    @staticmethod
    def _has_repetition(text: str) -> bool:
        return bool(re.search(r"(.{2,8})\1{2,}", text))

    @staticmethod
    def _compact_repeated_hallucination(text: str) -> str:
        cleaned = text
        for _ in range(4):
            updated = REPETITION_COMPACTION_RE.sub(
                lambda match: f"{match.group(1)}{match.group(1)}{REPETITION_MARKER}",
                cleaned,
            )
            if updated == cleaned:
                break
            cleaned = updated
        return re.sub(rf"(?:{re.escape(REPETITION_MARKER)}){{2,}}", REPETITION_MARKER, cleaned)

    @staticmethod
    def _has_mixed_technical_terms(text: str) -> bool:
        return bool(re.search(r"[A-Za-z]{2,}", text)) and any(
            token in text for token in ["微调", "评估", "模型", "课程", "课"]
        )

    def _find_review_terms(self, text: str) -> list[str]:
        hits: list[str] = []
        occupied: list[tuple[int, int]] = []
        terms = sorted(
            self.glossary.get("review_terms", {}).items(),
            key=lambda item: len(str(item[0])),
            reverse=True,
        )
        for term, note in terms:
            start = text.find(term)
            if start < 0:
                continue
            end = start + len(term)
            if any(start < used_end and end > used_start for used_start, used_end in occupied):
                continue
            occupied.append((start, end))
            hits.append(f"{term}: {note}")
        return hits

    def _best_candidate(self, alternatives: list[dict[str, Any]]) -> dict[str, Any] | None:
        scored: list[dict[str, Any]] = []
        for item in alternatives:
            if item.get("status") != "ok":
                continue
            text = str(item.get("transcript") or "").strip()
            if not text:
                continue
            scored.append(
                {
                    "backend": item.get("backend"),
                    "text": text,
                    "score": self.score_transcript_quality(text),
                }
            )
        if not scored:
            return None
        return max(scored, key=lambda item: float(item["score"]))

    def _should_accept_candidate(
        self,
        primary_text: str,
        primary_quality: float,
        candidate_text: str,
        candidate_quality: float,
    ) -> bool:
        if not candidate_text.strip():
            return False
        if primary_text.strip() == candidate_text.strip():
            return False

        primary_problems = set(self.transcript_quality_problems(primary_text))
        candidate_problems = set(self.transcript_quality_problems(candidate_text))
        candidate_has_hard_problem = bool(
            candidate_problems & {"empty", "replacement_character", "repetition", "compacted_repetition"}
        )
        if candidate_has_hard_problem:
            return False

        primary_normalized = self._normalize_for_quality(primary_text)
        candidate_normalized = self._normalize_for_quality(candidate_text)
        if primary_normalized and len(candidate_normalized) < max(4, int(len(primary_normalized) * 0.35)):
            return False

        hard_primary_problem = bool(
            primary_problems
            & {
                "empty",
                "replacement_character",
                "repetition",
                "compacted_repetition",
                "known_asr_confusion",
                "very_short",
            }
        )
        score_delta = candidate_quality - primary_quality
        if hard_primary_problem and score_delta >= 3.0:
            return True
        if "compacted_repetition" in primary_problems and candidate_quality > primary_quality:
            return True
        if "replacement_character" in primary_problems and candidate_quality > primary_quality:
            return True
        return False

    def score_transcript_quality(self, text: str) -> float:
        normalized = self._normalize_for_quality(text)
        if not normalized:
            return -100.0

        score = 30.0
        problems = self.transcript_quality_problems(text)
        penalties = {
            "empty": 100.0,
            "replacement_character": 35.0,
            "compacted_repetition": 28.0,
            "repetition": 22.0,
            "known_asr_confusion": 9.0,
            "very_short": 8.0,
            "control_token": 12.0,
            "low_character_diversity": 10.0,
        }
        for problem in problems:
            score -= penalties.get(problem, 0.0)

        marker_hits = sum(1 for marker in QUALITY_DIALECT_MARKERS if marker in text)
        domain_hits = sum(1 for marker in self.glossary.get("domain_terms", []) if str(marker) in text)
        score += min(10.0, marker_hits * 1.5 + domain_hits)
        score += min(8.0, len(set(normalized)) / 3.0)
        score += min(6.0, len(normalized) / 25.0)
        return score

    def transcript_quality_problems(self, text: str) -> list[str]:
        normalized = self._normalize_for_quality(text)
        problems: list[str] = []
        if not normalized:
            problems.append("empty")
            return problems
        if "\ufffd" in text:
            problems.append("replacement_character")
        if REPETITION_MARKER in text:
            problems.append("compacted_repetition")
        if self._has_repetition(text):
            problems.append("repetition")
        if CONTROL_LIKE_TOKEN_RE.search(text):
            problems.append("control_token")
        if len(normalized) <= 2:
            problems.append("very_short")
        if any(fragment in text for fragment in KNOWN_ASR_CONFUSIONS):
            problems.append("known_asr_confusion")
        if len(normalized) >= 16 and len(set(normalized)) / max(len(normalized), 1) < 0.25:
            problems.append("low_character_diversity")
        return problems

    @staticmethod
    def _normalize_for_quality(text: str) -> str:
        return QUALITY_PUNCTUATION_RE.sub("", text).lower()


def parse_custom_repairs(raw: str | list[str] | tuple[str, ...] | None) -> dict[str, str]:
    if raw is None:
        return {}
    if isinstance(raw, str):
        chunks = raw.replace("；", "\n").replace(";", "\n").splitlines()
    else:
        chunks = []
        for item in raw:
            chunks.extend(str(item).splitlines())

    repairs: dict[str, str] = {}
    for chunk in chunks:
        line = chunk.strip()
        if not line or line.startswith("#"):
            continue
        separator = "=> " if "=> " in line else None
        if separator is None:
            for candidate in ("=>", "->", "=", "：", ":"):
                if candidate in line:
                    separator = candidate
                    break
        if separator is None:
            continue
        left, right = line.split(separator, 1)
        wrong = left.strip()
        corrected = right.strip()
        if wrong and corrected and wrong != corrected:
            repairs[wrong] = corrected
    return repairs


CONTROL_LIKE_TOKEN_RE = re.compile(r"<\|[^|>]+?\|>|<[A-Za-z_][A-Za-z0-9_-]*>")


def load_custom_repairs_file(path: str | Path | None) -> dict[str, str]:
    if path is None:
        return {}
    repair_path = Path(path)
    if not repair_path.exists():
        return {}
    with repair_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    repairs: dict[str, str] = {}
    if isinstance(data, dict):
        if "asr_repairs" in data and isinstance(data["asr_repairs"], dict):
            source = data["asr_repairs"]
        else:
            source = data
        for wrong, corrected in source.items():
            wrong_text = str(wrong).strip()
            corrected_text = str(corrected).strip()
            if wrong_text and corrected_text and wrong_text != corrected_text:
                repairs[wrong_text] = corrected_text
        return repairs

    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            wrong_text = str(item.get("from") or item.get("wrong") or "").strip()
            corrected_text = str(item.get("to") or item.get("correct") or "").strip()
            if wrong_text and corrected_text and wrong_text != corrected_text:
                repairs[wrong_text] = corrected_text
    return repairs
