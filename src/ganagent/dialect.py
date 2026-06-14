from __future__ import annotations

from ganagent.models import DialectSignal


SHANGHAINESE_MARKERS = {
    "阿拉": 1.0,
    "侬": 1.0,
    "伊拉": 0.9,
    "吾": 1.0,
    "搿个": 1.0,
    "迭个": 1.0,
    "勿": 0.8,
    "伐": 0.7,
    "啥": 0.6,
    "哪能": 0.8,
    "辰光": 0.8,
    "交关": 0.8,
    "辣辣": 0.8,
    "等歇": 0.7,
    "灵": 0.5,
}


class ShanghaiDialectDetector:
    """Transparent rule-based Shanghainese/Wu signal detector."""

    def __init__(self, markers: dict[str, float] | None = None) -> None:
        self.markers = markers or SHANGHAINESE_MARKERS

    def detect(self, text: str) -> DialectSignal:
        hits: list[str] = []
        score = 0.0
        for marker, weight in self.markers.items():
            if marker in text:
                hits.append(marker)
                score += weight
        normalized = min(score / 4.0, 1.0)
        label = "shanghainese_or_wu" if normalized >= 0.25 else "unknown_or_mandarin"
        return DialectSignal(label=label, score=round(normalized, 3), markers=hits)


# Backward-compatible import for existing integrations.
GanDialectDetector = ShanghaiDialectDetector
GAN_MARKERS = SHANGHAINESE_MARKERS
