from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Segment:
    start: float
    end: float
    text: str
    confidence: float | None = None
    backend: str = "unknown"
    language_hint: str | None = None
    repaired_text: str | None = None

    def display_text(self) -> str:
        return self.repaired_text if self.repaired_text is not None else self.text


@dataclass
class Suspicion:
    segment_index: int
    severity: str
    reason: str
    evidence: str
    suggestion: str | None = None


@dataclass
class DialectSignal:
    label: str
    score: float
    markers: list[str] = field(default_factory=list)


@dataclass
class AgentResult:
    audio_path: str | None
    dialect: DialectSignal
    segments: list[Segment]
    suspicions: list[Suspicion]
    transcript: str
    mandarin_translation: str
    repairs: list[dict[str, Any]] = field(default_factory=list)
    alternatives: list[dict[str, Any]] = field(default_factory=list)
    agent_trace: list[dict[str, Any]] = field(default_factory=list)
    active_learning_items: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
