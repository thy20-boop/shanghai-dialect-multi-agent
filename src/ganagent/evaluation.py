from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Iterable


PUNCT_RE = re.compile(r"[\s，。！？、,.!?;；:：\"'“”‘’（）()\[\]{}<>《》-]+")


@dataclass
class EvaluationSummary:
    sample_count: int
    cer: float
    term_recall: float
    dialect_marker_recall: float
    exact_match_rate: float

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def normalize_text(text: str) -> str:
    return PUNCT_RE.sub("", text).lower()


def levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (0 if char_a == char_b else 1)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def cer(reference: str, hypothesis: str) -> float:
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return levenshtein_distance(ref, hyp) / len(ref)


def recall_for_items(reference: str, hypothesis: str, items: Iterable[str]) -> tuple[int, int]:
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)
    hit = 0
    total = 0
    for item in items:
        normalized_item = normalize_text(item)
        if not normalized_item:
            continue
        if normalized_item in ref:
            total += 1
            if normalized_item in hyp:
                hit += 1
    return hit, total


def evaluate_pairs(
    pairs: Iterable[tuple[str, str]],
    domain_terms: Iterable[str] = (),
    dialect_markers: Iterable[str] = (),
) -> EvaluationSummary:
    rows = list(pairs)
    if not rows:
        return EvaluationSummary(
            sample_count=0,
            cer=0.0,
            term_recall=0.0,
            dialect_marker_recall=0.0,
            exact_match_rate=0.0,
        )

    total_ref_len = 0
    total_distance = 0
    exact = 0
    term_hit = 0
    term_total = 0
    marker_hit = 0
    marker_total = 0

    for reference, hypothesis in rows:
        ref = normalize_text(reference)
        hyp = normalize_text(hypothesis)
        total_ref_len += max(len(ref), 1)
        total_distance += levenshtein_distance(ref, hyp)
        if ref == hyp:
            exact += 1

        hit, total = recall_for_items(reference, hypothesis, domain_terms)
        term_hit += hit
        term_total += total

        hit, total = recall_for_items(reference, hypothesis, dialect_markers)
        marker_hit += hit
        marker_total += total

    return EvaluationSummary(
        sample_count=len(rows),
        cer=round(total_distance / total_ref_len, 4),
        term_recall=round(term_hit / term_total, 4) if term_total else 0.0,
        dialect_marker_recall=round(marker_hit / marker_total, 4) if marker_total else 0.0,
        exact_match_rate=round(exact / len(rows), 4),
    )
