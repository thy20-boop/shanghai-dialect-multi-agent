from __future__ import annotations

import html
from typing import Any


def render_html_dashboard(
    predictions: list[dict[str, Any]],
    evaluation: dict[str, Any],
    title: str = "Shanghai Dialect ASR Dashboard",
) -> str:
    quality = summarize_predictions(predictions)
    cards = [
        ("Samples", evaluation.get("sample_count", len(predictions))),
        ("CER", evaluation.get("cer", "n/a")),
        ("Term Recall", evaluation.get("term_recall", "n/a")),
        ("Dialect Recall", evaluation.get("dialect_marker_recall", "n/a")),
        ("Exact Match", evaluation.get("exact_match_rate", "n/a")),
        ("Avg Repairs", quality["avg_repairs"]),
        ("Avg Suspicions", quality["avg_suspicions"]),
        ("Learning Items", quality["active_learning_items"]),
        ("Needs Review", quality["needs_review"]),
    ]
    card_html = "\n".join(
        f'<section class="metric"><span>{html.escape(str(label))}</span><strong>{html.escape(str(value))}</strong></section>'
        for label, value in cards
    )

    rows = "\n".join(render_prediction_row(index, row) for index, row in enumerate(predictions))
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #202124;
      --muted: #5f6368;
      --line: #dadce0;
      --panel: #f8fafd;
      --accent: #146c5f;
      --warn: #9a6700;
      --danger: #b3261e;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", "Microsoft YaHei", Arial, sans-serif;
      color: var(--ink);
      background: #ffffff;
    }}
    header {{
      padding: 28px 32px 18px;
      border-bottom: 1px solid var(--line);
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 26px;
      letter-spacing: 0;
      font-weight: 650;
    }}
    .subtitle {{
      margin: 0;
      color: var(--muted);
      font-size: 14px;
    }}
    main {{
      padding: 22px 32px 36px;
      max-width: 1180px;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 12px;
      margin-bottom: 22px;
    }}
    .metric {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px 16px;
      background: var(--panel);
      min-height: 78px;
    }}
    .metric span {{
      display: block;
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 10px;
    }}
    .metric strong {{
      font-size: 24px;
      font-weight: 650;
    }}
    .sample {{
      border-top: 1px solid var(--line);
      padding: 18px 0;
    }}
    .sample h2 {{
      margin: 0 0 10px;
      font-size: 17px;
      letter-spacing: 0;
    }}
    .meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 12px;
    }}
    .pill {{
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 4px 9px;
      font-size: 12px;
      color: var(--muted);
      background: #fff;
    }}
    .pill.warn {{ color: var(--warn); border-color: #f1c56b; }}
    .pill.danger {{ color: var(--danger); border-color: #ef9a9a; }}
    .grid {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      gap: 12px;
    }}
    .box {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      min-height: 96px;
    }}
    .box h3 {{
      margin: 0 0 8px;
      font-size: 13px;
      color: var(--muted);
      font-weight: 600;
    }}
    .box p {{
      margin: 0;
      line-height: 1.55;
      font-size: 15px;
    }}
    @media (max-width: 760px) {{
      header, main {{ padding-left: 18px; padding-right: 18px; }}
      .grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(title)}</h1>
    <p class="subtitle">上海话/吴语多智能体 agent 的批处理结果、指标、候选分歧和主动学习概览。</p>
  </header>
  <main>
    <section class="metrics">{card_html}</section>
    <section>{rows or "<p>No predictions.</p>"}</section>
  </main>
</body>
</html>
"""


def render_prediction_row(index: int, row: dict[str, Any]) -> str:
    markers = ", ".join(str(item) for item in row.get("dialect_markers", [])) or "none"
    suspicion_count = int(row.get("suspicion_count", 0) or 0)
    repair_count = int(row.get("repair_count", 0) or 0)
    active_learning_count = len(row.get("active_learning_items", []) or [])
    agent_count = len(row.get("agent_trace", []) or [])
    suspicion_class = "danger" if suspicion_count >= 2 else "warn" if suspicion_count else ""
    transcript = html.escape(str(row.get("transcript", "")))
    translation = html.escape(str(row.get("mandarin_translation", "")))
    reference = html.escape(str(row.get("reference", "") or ""))
    audio = html.escape(str(row.get("audio", "") or ""))
    return f"""
<article class="sample">
  <h2>Sample {html.escape(str(row.get("id", index)))}</h2>
  <div class="meta">
    <span class="pill">audio: {audio or "n/a"}</span>
    <span class="pill">dialect: {html.escape(str(row.get("dialect", "n/a")))}</span>
    <span class="pill">markers: {html.escape(markers)}</span>
    <span class="pill">repairs: {repair_count}</span>
    <span class="pill {suspicion_class}">suspicions: {suspicion_count}</span>
    <span class="pill">agents: {agent_count}</span>
    <span class="pill warn">learning: {active_learning_count}</span>
  </div>
  <div class="grid">
    <section class="box"><h3>Reference</h3><p>{reference or "n/a"}</p></section>
    <section class="box"><h3>Transcript</h3><p>{transcript or "n/a"}</p></section>
    <section class="box"><h3>Mandarin Translation</h3><p>{translation or "n/a"}</p></section>
    <section class="box"><h3>Repairs</h3><p>{html.escape(render_repairs(row))}</p></section>
    <section class="box"><h3>Agent Trace</h3><p>{html.escape(render_agent_trace(row))}</p></section>
    <section class="box"><h3>Learning Reasons</h3><p>{html.escape(render_learning_reasons(row))}</p></section>
  </div>
</article>
"""


def render_repairs(row: dict[str, Any]) -> str:
    repairs = row.get("repairs", [])
    if not repairs:
        return "none"
    parts: list[str] = []
    for repair in repairs:
        replacements = repair.get("replacements") or []
        if replacements:
            detail = ", ".join(
                f"{item.get('from')}->{item.get('to')} x{item.get('count', 1)}"
                for item in replacements
            )
            parts.append(f"{repair.get('type', 'repair')}: {detail}")
        else:
            parts.append(f"{repair.get('type', 'repair')}: {repair.get('original')} -> {repair.get('repaired')}")
    return " | ".join(parts)


def render_agent_trace(row: dict[str, Any]) -> str:
    trace = row.get("agent_trace", []) or []
    if not trace:
        return "none"
    return " -> ".join(
        f"{item.get('agent', 'agent')}[{item.get('status', 'unknown')}]"
        for item in trace
    )


def render_learning_reasons(row: dict[str, Any]) -> str:
    items = row.get("active_learning_items", []) or []
    if not items:
        return "none"
    reasons: list[str] = []
    for item in items:
        reason = item.get("reason", [])
        if isinstance(reason, list):
            reasons.extend(str(value) for value in reason)
        else:
            reasons.append(str(reason))
    return ", ".join(sorted(set(reasons))) or "none"


def summarize_predictions(predictions: list[dict[str, Any]]) -> dict[str, int | float]:
    if not predictions:
        return {
            "avg_repairs": 0.0,
            "avg_suspicions": 0.0,
            "active_learning_items": 0,
            "needs_review": 0,
        }
    repair_total = sum(int(row.get("repair_count", 0) or 0) for row in predictions)
    suspicion_total = sum(int(row.get("suspicion_count", 0) or 0) for row in predictions)
    learning_total = sum(len(row.get("active_learning_items", []) or []) for row in predictions)
    needs_review = sum(
        1
        for row in predictions
        if int(row.get("suspicion_count", 0) or 0) > 0 or row.get("active_learning_items")
    )
    count = len(predictions)
    return {
        "avg_repairs": round(repair_total / count, 2),
        "avg_suspicions": round(suspicion_total / count, 2),
        "active_learning_items": learning_total,
        "needs_review": needs_review,
    }
