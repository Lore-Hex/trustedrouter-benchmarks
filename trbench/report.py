"""Shared report rendering: a ranked markdown table + a flat SVG bar chart.

Kept deliberately generic so every eval renders the same way. Charts use the
same flat, two-color style as PrometheusBench (no gradients/glow).
"""
from __future__ import annotations

import html
from typing import Any

FONT = 'font-family="Inter, Arial, sans-serif"'


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.1f}"
    return str(v)


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    """columns = list of (header, key). A leading Rank column is added."""
    head = "| Rank | " + " | ".join(h for h, _ in columns) + " |"
    sep = "|---:|" + "|".join("---:" if k != "model" else "---" for _, k in columns) + "|"
    lines = [head, sep]
    for i, row in enumerate(rows, start=1):
        cells = []
        for _, key in columns:
            val = row.get(key, "")
            cells.append(f"`{val}`" if key == "model" else _fmt(val))
        lines.append(f"| {i} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def svg_bar_chart(
    rows: list[dict[str, Any]],
    *,
    score_key: str,
    max_score: float,
    title: str,
    subtitle: str,
    label_suffix: str = "",
) -> str:
    row_h = 27
    top = 52
    left = 300
    width = 960
    height = top + row_h * len(rows) + 52
    max_bar = 380
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="24" y="30" {FONT} font-size="22" font-weight="700" fill="#111827">{html.escape(title)}</text>',
        f'<text x="24" y="50" {FONT} font-size="12" fill="#4b5563">{html.escape(subtitle)}</text>',
    ]
    for i, row in enumerate(rows):
        y = top + i * row_h
        score = float(row.get(score_key) or 0)
        frac = max(0.0, min(1.0, score / max_score if max_score else 0))
        bar_w = max(1, int(max_bar * frac))
        color = "#2563eb" if frac >= 0.66 else "#f97316" if frac >= 0.33 else "#dc2626"
        model = html.escape(str(row.get("model", "")))
        val = f"{score:.1f}".rstrip("0").rstrip(".") + label_suffix
        parts.extend(
            [
                f'<text x="24" y="{y + 17}" {FONT} font-size="12" fill="#111827">{i + 1}. {model}</text>',
                f'<rect x="{left}" y="{y + 4}" width="{max_bar}" height="16" rx="4" fill="#e5e7eb"/>',
                f'<rect x="{left}" y="{y + 4}" width="{bar_w}" height="16" rx="4" fill="{color}"/>',
                f'<text x="{left + max_bar + 12}" y="{y + 17}" {FONT} '
                f'font-size="12" font-weight="700" fill="#111827">{val}</text>',
            ]
        )
    parts.append("</svg>")
    return "\n".join(parts)


def splice_readme(readme_text: str, marker: str, block: str) -> str:
    start = f"<!-- {marker}_START -->"
    end = f"<!-- {marker}_END -->"
    if start not in readme_text or end not in readme_text:
        readme_text = readme_text.rstrip() + f"\n\n{start}\n{end}\n"
    before, rest = readme_text.split(start, 1)
    _old, after = rest.split(end, 1)
    return before.rstrip() + "\n\n" + start + "\n\n" + block + "\n\n" + end + after
