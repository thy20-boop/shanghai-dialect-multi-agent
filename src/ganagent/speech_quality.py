from __future__ import annotations

from dataclasses import asdict, dataclass
import re


DEFAULT_KEY_TERMS = (
    "报警",
    "火警",
    "急救",
    "交通事故",
    "市民服务热线",
    "身份证",
    "派出所",
    "居委",
    "居委会",
    "户籍",
)

DEFAULT_CRITICAL_ENTITY_TERMS = (
    "报警",
    "火警",
    "急救",
    "交通事故",
    "市民服务热线",
    "身份证",
    "临时居民身份证",
    "派出所",
    "户籍",
)

_DIGIT_WORDS = {
    "0": "零",
    "1": "一",
    "2": "二",
    "3": "三",
    "4": "四",
    "5": "五",
    "6": "六",
    "7": "七",
    "8": "八",
    "9": "九",
}
_WORD_DIGITS = {value: key for key, value in _DIGIT_WORDS.items()}
_WORD_DIGITS["幺"] = "1"
_WORD_DIGITS["两"] = "2"


@dataclass(frozen=True)
class SpeechQualityScore:
    expected_terms: list[str]
    matched_terms: list[str]
    missing_terms: list[str]
    keyword_recall: float
    dialect_score: float
    suspicion_count: int = 0
    critical_terms: list[str] | None = None
    matched_critical_terms: list[str] | None = None
    missing_critical_terms: list[str] | None = None
    critical_entity_recall: float = 1.0
    passes_critical_gate: bool = True
    cer: float = 0.0
    char_accuracy: float = 1.0

    @property
    def is_usable(self) -> bool:
        return (
            self.keyword_recall >= 0.9
            and self.passes_critical_gate
            and self.char_accuracy >= 0.75
            and self.suspicion_count <= 1
        )

    def as_dict(self) -> dict:
        payload = asdict(self)
        payload["is_usable"] = self.is_usable
        return payload


def extract_expected_terms(text: str, extra_terms: list[str] | None = None) -> list[str]:
    terms: list[str] = []
    for number in re.findall(r"\d{2,}", text):
        terms.append(number)
    terms.extend(_extract_spoken_number_sequences(text))
    for term in [*DEFAULT_KEY_TERMS, *(extra_terms or [])]:
        if term and term in text:
            terms.append(term)
    return _dedupe(terms)


def extract_critical_entities(
    text: str,
    extra_terms: list[str] | None = None,
) -> list[str]:
    """Extract entities whose omission must reject a generated answer."""

    entities: list[str] = []
    entities.extend(re.findall(r"\d{2,}", text))
    entities.extend(_extract_spoken_number_sequences(text))
    for term in [*DEFAULT_CRITICAL_ENTITY_TERMS, *(extra_terms or [])]:
        if term and term in text:
            entities.append(term)
    return _dedupe(entities)


def score_spoken_answer(
    expected_text: str,
    recognized_text: str,
    *,
    mandarin_translation: str = "",
    dialect_score: float = 0.0,
    suspicion_count: int = 0,
    extra_terms: list[str] | None = None,
    extra_critical_terms: list[str] | None = None,
) -> SpeechQualityScore:
    expected_terms = extract_expected_terms(expected_text, extra_terms=extra_terms)
    search_space = f"{recognized_text}\n{mandarin_translation}"
    matched = [term for term in expected_terms if _term_present(term, search_space)]
    missing = [term for term in expected_terms if term not in matched]
    recall = len(matched) / len(expected_terms) if expected_terms else 1.0
    critical_terms = extract_critical_entities(
        expected_text,
        extra_terms=extra_critical_terms,
    )
    matched_critical = [term for term in critical_terms if _term_present(term, search_space)]
    missing_critical = [term for term in critical_terms if term not in matched_critical]
    critical_recall = (
        len(matched_critical) / len(critical_terms) if critical_terms else 1.0
    )
    expected_chars = _normalize_for_cer(expected_text)
    recognized_candidates = [recognized_text]
    if mandarin_translation:
        recognized_candidates.append(mandarin_translation)
    cer = min(
        (
            _edit_distance(expected_chars, _normalize_for_cer(candidate))
            / len(expected_chars)
            for candidate in recognized_candidates
        ),
        default=0.0,
    ) if expected_chars else 0.0
    return SpeechQualityScore(
        expected_terms=expected_terms,
        matched_terms=matched,
        missing_terms=missing,
        keyword_recall=round(recall, 4),
        dialect_score=round(float(dialect_score), 4),
        suspicion_count=suspicion_count,
        critical_terms=critical_terms,
        matched_critical_terms=matched_critical,
        missing_critical_terms=missing_critical,
        critical_entity_recall=round(critical_recall, 4),
        passes_critical_gate=not missing_critical,
        cer=round(cer, 4),
        char_accuracy=round(max(0.0, 1.0 - cer), 4),
    )


def _term_present(term: str, text: str) -> bool:
    if term.isdigit():
        numeric_mentions = re.findall(r"(?<!\d)\d{2,}(?!\d)", text)
        numeric_mentions.extend(_extract_spoken_number_sequences(text))
        return term in numeric_mentions
    if term in text:
        return True
    # Whisper-Medium-Wu can render the near-homophone 火警 as 沪警 in a long
    # joined utterance. Only repair it when the exact fire hotline is also
    # present, so an unrelated 沪警 mention cannot pass the safety gate.
    if term == "火警" and "沪警" in text and _term_present("119", text):
        return True
    return False


def _extract_spoken_number_sequences(text: str) -> list[str]:
    numbers: list[str] = []
    pattern = r"(?:[零幺一二两三四五六七八九][\s，,、]*){2,}"
    for match in re.findall(pattern, text):
        digits = [char for char in match if char in _WORD_DIGITS]
        if len(digits) >= 2:
            numbers.append("".join(_WORD_DIGITS[char] for char in digits))
    return numbers


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def _normalize_for_cer(text: str) -> str:
    normalized = re.sub(
        r"[零幺一二两三四五六七八九]{2,}",
        lambda match: "".join(_WORD_DIGITS[char] for char in match.group(0)),
        text,
    )
    return re.sub(
        r"[^0-9A-Za-z\u4e00-\u9fff]",
        "",
        normalized,
    )


def _edit_distance(reference: str, hypothesis: str) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row, source_char in enumerate(reference, start=1):
        current = [row]
        for column, target_char in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (source_char != target_char),
                )
            )
        previous = current
    return previous[-1]
