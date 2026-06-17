from __future__ import annotations

import re
from dataclasses import dataclass, field

from ganagent.tts import WuReferenceExpert, allows_prefix_trim


SHANGHAI_GUARD_WU_NAME = "ShanghaiGuard-Wu"
DEFAULT_COSY_SPEEDS = (0.94, 0.88, 1.0, 0.82, 0.9)
DEFAULT_CANDIDATE_SEEDS = (1986, 2026, 3407, 42, 8675309)


@dataclass(frozen=True)
class WuGenerationProfile:
    generator: str
    url: str
    reference: WuReferenceExpert | None
    speed: float
    seed: int

    def as_dict(self) -> dict:
        return {
            "generator": self.generator,
            "url": self.url,
            "reference_expert": self.reference.expert_id if self.reference else "server_default",
            "speed": self.speed,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class WuGenerationPolicy:
    expert_name: str
    task_type: str
    prefix_trim_enabled: bool
    reference_experts: tuple[WuReferenceExpert, ...] = field(default_factory=tuple)
    endpoints: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    schedule: tuple[WuGenerationProfile, ...] = field(default_factory=tuple)
    minimum_reference_exploration: int = 1
    critical_entity_gate: bool = True

    def as_dict(self) -> dict:
        return {
            "expert_name": self.expert_name,
            "task_type": self.task_type,
            "prefix_trim_enabled": self.prefix_trim_enabled,
            "reference_experts": [expert.expert_id for expert in self.reference_experts],
            "endpoints": [endpoint_id for endpoint_id, _ in self.endpoints],
            "schedule": [profile.as_dict() for profile in self.schedule],
            "minimum_reference_exploration": self.minimum_reference_exploration,
            "critical_entity_gate": self.critical_entity_gate,
        }


def classify_wu_generation_task(text: str) -> str:
    """Classify the spoken-answer risk profile for Wu speech generation."""

    if re.search(r"\b(110|119|120|12345)\b|报警|火警|急救|热线|电话|号码|求助", text):
        return "hotline"
    if any(term in text for term in ("身份证", "派出所", "户籍", "居住证", "社区", "办证")):
        return "public_service"
    return "general"


def build_wu_generation_policy(
    text: str,
    *,
    reference_experts: list[WuReferenceExpert] | tuple[WuReferenceExpert, ...],
    endpoints: list[tuple[str, str]] | tuple[tuple[str, str], ...],
    no_prefix_trim: bool = False,
    speeds: tuple[float, ...] = DEFAULT_COSY_SPEEDS,
    seeds: tuple[int, ...] = DEFAULT_CANDIDATE_SEEDS,
    max_candidates: int = 8,
) -> WuGenerationPolicy:
    """Build the project-owned generation policy around the Wu acoustic expert.

    The public CosyVoice2-Wu checkpoint provides the base acoustic skill. This
    layer is ours: it decides which prompt expert can be trusted, which seeds
    must be explored, when prefix trimming is allowed, and how a second endpoint
    can compete without replacing the primary model.
    """

    task_type = classify_wu_generation_task(text)
    experts = tuple(_filter_reference_experts(task_type, tuple(reference_experts)))
    endpoints_tuple = tuple(endpoints)
    baseline_reference = experts[0] if experts else None
    primary_endpoint = _endpoint_url(endpoints_tuple, "primary")
    secondary_endpoint = _endpoint_url(endpoints_tuple, "secondary")

    baseline_schedule = tuple(
        WuGenerationProfile(
            generator="primary",
            url=primary_endpoint,
            reference=baseline_reference,
            speed=speeds[index],
            seed=seeds[index],
        )
        for index in range(min(len(speeds), len(seeds)))
    )
    schedule: list[WuGenerationProfile] = list(baseline_schedule[:2])

    if task_type != "hotline":
        schedule.extend(
            WuGenerationProfile(
                generator="primary",
                url=primary_endpoint,
                reference=reference,
                speed=speeds[0],
                seed=seeds[0],
            )
            for reference in experts[1:]
        )

    if secondary_endpoint and task_type != "hotline":
        schedule.append(
            WuGenerationProfile(
                generator="secondary",
                url=secondary_endpoint,
                reference=baseline_reference,
                speed=speeds[0],
                seed=seeds[0],
            )
        )

    schedule.extend(baseline_schedule[2:])
    schedule = schedule[:max_candidates]

    prefix_trim_enabled = (
        not no_prefix_trim
        and task_type != "hotline"
        and allows_prefix_trim(text)
    )
    minimum_reference_exploration = min(3, len(schedule)) if task_type != "hotline" else 1
    return WuGenerationPolicy(
        expert_name=SHANGHAI_GUARD_WU_NAME,
        task_type=task_type,
        prefix_trim_enabled=prefix_trim_enabled,
        reference_experts=experts,
        endpoints=endpoints_tuple,
        schedule=tuple(schedule),
        minimum_reference_exploration=minimum_reference_exploration,
    )


def _endpoint_url(endpoints: tuple[tuple[str, str], ...], endpoint_id: str) -> str:
    for current_id, url in endpoints:
        if current_id == endpoint_id:
            return url
    return endpoints[0][1] if endpoints else ""


def _filter_reference_experts(
    task_type: str,
    reference_experts: tuple[WuReferenceExpert, ...],
) -> tuple[WuReferenceExpert, ...]:
    if task_type != "hotline":
        return reference_experts
    certified = tuple(
        expert
        for expert in reference_experts
        if "hotline" in expert.domains
    )
    return certified or reference_experts[:1]
