from __future__ import annotations

from ganagent.models import AgentResult
from ganagent.product import build_translation_product


def render_markdown_report(result: AgentResult, title: str = "Shanghai Dialect ASR Report") -> str:
    dialect = result.dialect
    product = build_translation_product(result)
    lines = [
        f"# {title}",
        "",
        "## Dialect Signal",
        "",
        f"- Label: `{dialect.label}`",
        f"- Score: `{dialect.score}`",
        f"- Markers: {', '.join(dialect.markers) if dialect.markers else 'none'}",
        "",
        "## Transcript",
        "",
        result.transcript or "(empty)",
        "",
        "## Mandarin Translation",
        "",
        result.mandarin_translation or "(empty)",
        "",
        "## Quality Assessment",
        "",
        f"- Status: `{product.status}` ({product.status_label})",
        f"- Quality score: `{product.quality_score}`",
        f"- Consensus score: `{product.consensus_score if product.consensus_score is not None else 'n/a'}`",
        f"- Suggested action: {product.action_suggestion or 'none'}",
        "",
        "## Multi-Agent Trace",
        "",
    ]

    if result.agent_trace:
        for step in result.agent_trace:
            agent = step.get("agent", "agent")
            role = step.get("role", "role")
            status = step.get("status", "unknown")
            summary = step.get("summary", "")
            lines.append(f"- **{agent}** ({role}, `{status}`): {summary}")
    else:
        lines.append("- none")

    lines.extend(["", "## ASR Alternatives", ""])
    if result.alternatives:
        for item in result.alternatives:
            backend = item.get("backend", "unknown")
            status = item.get("status", "unknown")
            transcript = item.get("transcript") or item.get("error") or ""
            lines.append(f"- `{backend}` `{status}`: {transcript or '(empty)'}")
    else:
        lines.append("- none")

    lines.extend(["", "## Active Learning Candidates", ""])
    if result.active_learning_items:
        for item in result.active_learning_items:
            reason = ", ".join(str(value) for value in item.get("reason", []))
            transcript = item.get("primary_transcript", "")
            lines.append(f"- Reasons: `{reason or 'unknown'}`; transcript: {transcript or '(empty)'}")
    else:
        lines.append("- none")

    lines.extend(["", "## Repairs", ""])
    if result.repairs:
        for repair in result.repairs:
            segment_index = repair.get("segment_index", "all")
            lines.append(
                f"- `{repair.get('type', 'repair')}` segment {segment_index}: "
                f"`{repair.get('original', '')}` -> `{repair.get('repaired', '')}`"
            )
            for replacement in repair.get("replacements", []):
                count = replacement.get("count", 1)
                source = replacement.get("source", "rule")
                lines.append(
                    f"  - {replacement.get('from')} -> {replacement.get('to')} "
                    f"(count={count}, source={source})"
                )
    else:
        lines.append("- none")

    lines.extend(["", "## Suspicious Segments", ""])
    if result.suspicions:
        for suspicion in result.suspicions:
            lines.append(
                f"- `{suspicion.severity}` segment {suspicion.segment_index}: "
                f"{suspicion.reason}; evidence: `{suspicion.evidence}`"
            )
            if suspicion.suggestion:
                lines.append(f"  - Suggestion: {suspicion.suggestion}")
    else:
        lines.append("- none")

    return "\n".join(lines) + "\n"
