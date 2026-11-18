"""Text, JSON, and self-contained HTML reports.

The HTML file has no external URLs, scripts, or images. Open it from disk.
"""

from __future__ import annotations

import html
import json

from annotrust.results import AnnotatorReport, ItemReport, PairReport, ScanResult

_TEXT_ANNOTATORS = 40
_TEXT_ITEMS = 12
_HTML_ITEMS = 100


def render_text(result: ScanResult) -> str:
    """Return the terminal report, ending in a newline."""
    counts = result.severity_counts()
    lines = [
        f"Annotrust {result.version}",
        f"source: {result.source or 'dataframe'}",
        (
            f"annotations: {result.n_annotations}   items: {result.n_items}   "
            f"annotators: {result.n_annotators}   classes: {', '.join(result.classes)}"
        ),
        f"consensus: Dawid-Skene EM, {result.iterations} iterations, log-likelihood {result.log_likelihood:.1f}",
        (f"annotator severity: {counts['high']} high, {counts['medium']} medium, {counts['low']} low"),
        "",
        f"{'id':<16} {'primary':<13} {'severity':<8} {'n':>5} {'accuracy':>8}  notes",
    ]
    shown = result.annotators[:_TEXT_ANNOTATORS]
    for annotator in shown:
        lines.append(_annotator_line(annotator))
    hidden = len(result.annotators) - len(shown)
    if hidden:
        lines.append(f"... {hidden} more annotators in the JSON report")
    flagged = [pair for pair in result.pairs if pair.flagged]
    lines.extend(["", f"pairs scored: {len(result.pairs)}    flagged pairs: {len(flagged)}"])
    if flagged:
        lines.append(f"{'a':<16} {'b':<16} {'n':>5} {'kappa':>7} {'excess':>7} {'wrong':>6} {'score':>7}  severity")
        for pair in flagged:
            lines.append(_pair_line(pair))
    else:
        lines.append("flagged pairs: none")
    suspects = [item for item in result.items if item.suspect]
    suspects.sort(key=lambda item: (item.severity != "medium", -item.normalized_entropy, item.item_id))
    lines.extend(
        [
            "",
            f"item flags: {len(suspects)} of {result.n_items}",
            "Flagged items are unstable or the majority label conflicts with the Dawid-Skene consensus.",
            "That is a suspicion, not proof the item is mislabeled.",
        ]
    )
    if suspects:
        lines.append(f"{'item':<14} {'estimate':<12} {'conf':>6} {'majority':<12} {'share':>6} {'entropy':>7}  reasons")
        for item in suspects[:_TEXT_ITEMS]:
            lines.append(_item_line(item))
        if len(suspects) > _TEXT_ITEMS:
            lines.append(f"... {len(suspects) - _TEXT_ITEMS} more flagged items in the JSON report")
    if result.warnings:
        lines.extend(["", "warnings:"])
        lines.extend(f"- {warning}" for warning in result.warnings)
    lines.extend(
        [
            "",
            "Exit status: --fail-on looks at annotator severity only. Flagged items do not change it.",
            "",
        ]
    )
    return "\n".join(lines)


def render_json(result: ScanResult) -> str:
    """Return pretty JSON, ending in a newline."""
    return json.dumps(result.to_dict(), indent=2) + "\n"


def render_html(result: ScanResult) -> str:
    """Return one HTML document with CSS inlined and no external resources."""
    counts = result.severity_counts()
    suspects = [item for item in result.items if item.suspect]
    suspects.sort(key=lambda item: (item.severity != "medium", -item.normalized_entropy, item.item_id))
    flagged = [pair for pair in result.pairs if pair.flagged]
    parts = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>Annotrust report</title>",
        f"<style>{_CSS}</style>",
        "</head>",
        "<body>",
        "<header>",
        f'<p class="eyebrow">Annotrust {html.escape(result.version)}</p>',
        "<h1>Annotation integrity report</h1>",
        f'<p class="source">{html.escape(result.source or "dataframe")}</p>',
        "</header>",
        '<section class="stats">',
        _stat("Annotations", str(result.n_annotations)),
        _stat("Items", str(result.n_items)),
        _stat("Annotators", str(result.n_annotators)),
        _stat("High", str(counts["high"])),
        _stat("Medium", str(counts["medium"])),
        _stat("Low", str(counts["low"])),
        "</section>",
        "<section>",
        "<h2>Annotators</h2>",
        f'<p class="note">Classes, sorted: {html.escape(", ".join(result.classes))}. '
        f"Dawid-Skene ran {result.iterations} iterations. "
        "Confusion-matrix rows are the estimated true class and columns are the label submitted.</p>",
        _annotator_table(result),
        "</section>",
        "<section>",
        "<h2>Colluding pairs</h2>",
        f'<p class="note">Pairs scored: {len(result.pairs)}. Flagged: {len(flagged)}.</p>',
        _pair_table(flagged),
        "</section>",
        "<section>",
        "<h2>Flagged items</h2>",
        '<p class="note">A flagged item is unstable, or the majority label conflicts with the consensus. '
        "That is a suspicion, not proof the item is mislabeled. "
        f"Showing {min(len(suspects), _HTML_ITEMS)} of {len(suspects)} flagged items "
        f"({result.n_items} items scored).</p>",
        _item_table(suspects[:_HTML_ITEMS]),
        "</section>",
    ]
    if result.warnings:
        parts.extend(["<section>", "<h2>Warnings</h2>", "<ul>"])
        parts.extend(f"<li>{html.escape(warning)}</li>" for warning in result.warnings)
        parts.extend(["</ul>", "</section>"])
    parts.extend(
        [
            "<footer>",
            "<p>Heuristic thresholds, not an audit. The benchmark numbers shipped with Annotrust "
            "come from the simulator, not from a production labeling vendor. "
            "Encord and Label Studio readers are unofficial best-effort parsers and are not endorsed "
            "by those projects. --fail-on looks at annotator severity only.</p>",
            "</footer>",
            "</body>",
            "</html>",
            "",
        ]
    )
    return "\n".join(parts)


def _annotator_line(annotator: AnnotatorReport) -> str:
    notes = "; ".join(flag.summary for flag in annotator.flags)
    severity = annotator.severity or "-"
    accuracy = "-" if annotator.accuracy is None else f"{annotator.accuracy:.3f}"
    return (
        f"{_cell(annotator.annotator_id, 16):<16} {_cell(annotator.primary, 13):<13} "
        f"{severity:<8} {annotator.n_labels:5d} {accuracy:>8}  {_cell(notes, 180)}"
    )


def _pair_line(pair: PairReport) -> str:
    return (
        f"{_cell(pair.annotator_a, 16):<16} {_cell(pair.annotator_b, 16):<16} {pair.overlap:5d} "
        f"{pair.kappa:7.2f} {pair.excess:7.2f} {pair.joint_wrong:6d} {pair.score:7.2f}  {pair.severity or '-'}"
    )


def _item_line(item: ItemReport) -> str:
    reasons = ",".join(item.reasons)
    return (
        f"{_cell(item.item_id, 14):<14} {_cell(item.estimated_label, 12):<12} {item.confidence:6.2f} "
        f"{_cell(item.majority_label, 12):<12} {item.majority_fraction:6.2f} {item.normalized_entropy:7.2f}  "
        f"{_cell(reasons, 80)}"
    )


def _cell(text: str, width: int) -> str:
    flat = " ".join(text.split())
    if len(flat) <= width:
        return flat
    if width <= 3:
        return flat[:width]
    return flat[: width - 3] + "..."


def _stat(label: str, value: str) -> str:
    return f'<div class="stat"><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong></div>'


def _badge(severity: str | None) -> str:
    if severity is None:
        return '<span class="badge ok">none</span>'
    return f'<span class="badge {html.escape(severity)}">{html.escape(severity)}</span>'


def _annotator_table(result: ScanResult) -> str:
    rows = [
        "<table>",
        "<thead><tr><th>Id</th><th>Primary</th><th>Severity</th><th>Labels</th>"
        "<th>Agreement</th><th>Entropy</th><th>Streak</th><th>Notes</th></tr></thead>",
        "<tbody>",
    ]
    for annotator in result.annotators:
        accuracy = "-" if annotator.accuracy is None else f"{annotator.accuracy:.3f}"
        entropy = "-" if annotator.normalized_entropy is None else f"{annotator.normalized_entropy:.2f}"
        notes = "; ".join(flag.summary for flag in annotator.flags) or "-"
        rows.append(
            "<tr>"
            f"<td>{html.escape(annotator.annotator_id)}</td>"
            f"<td>{html.escape(annotator.primary)}</td>"
            f"<td>{_badge(annotator.severity)}</td>"
            f"<td>{annotator.n_labels}</td>"
            f"<td>{accuracy}</td>"
            f"<td>{entropy}</td>"
            f"<td>{annotator.longest_streak}</td>"
            f"<td>{html.escape(notes)}</td>"
            "</tr>"
        )
        if annotator.flips or annotator.severity in {"high", "medium"}:
            rows.append('<tr><td colspan="8">' + _confusion_table(result.classes, annotator) + "</td></tr>")
    rows.extend(["</tbody>", "</table>"])
    return "\n".join(rows)


def _confusion_table(classes: tuple[str, ...], annotator: AnnotatorReport) -> str:
    header = "".join(f"<th>{html.escape(label)}</th>" for label in classes)
    body = []
    for label, row in zip(classes, annotator.confusion, strict=True):
        cells = "".join(f"<td>{value:.2f}</td>" for value in row)
        body.append(f"<tr><th>{html.escape(label)}</th>{cells}</tr>")
    return (
        '<table class="confusion"><caption>Confusion, estimated true class by submitted label, '
        f"{html.escape(annotator.annotator_id)}</caption>"
        f"<thead><tr><th>true / pred</th>{header}</tr></thead><tbody>{''.join(body)}</tbody></table>"
    )


def _pair_table(pairs: list[PairReport]) -> str:
    if not pairs:
        return "<p>No flagged pairs.</p>"
    rows = [
        "<table>",
        "<thead><tr><th>A</th><th>B</th><th>Overlap</th><th>Kappa</th><th>Excess</th>"
        "<th>Joint wrong</th><th>Score</th><th>Severity</th></tr></thead>",
        "<tbody>",
    ]
    for pair in pairs:
        rows.append(
            "<tr>"
            f"<td>{html.escape(pair.annotator_a)}</td>"
            f"<td>{html.escape(pair.annotator_b)}</td>"
            f"<td>{pair.overlap}</td>"
            f"<td>{pair.kappa:.2f}</td>"
            f"<td>{pair.excess:.2f}</td>"
            f"<td>{pair.joint_wrong}</td>"
            f"<td>{pair.score:.2f}</td>"
            f"<td>{_badge(pair.severity)}</td>"
            "</tr>"
        )
    rows.extend(["</tbody>", "</table>"])
    return "\n".join(rows)


def _item_table(items: list[ItemReport]) -> str:
    if not items:
        return "<p>No flagged items.</p>"
    rows = [
        "<table>",
        "<thead><tr><th>Item</th><th>Estimate</th><th>Confidence</th><th>Majority</th>"
        "<th>Share</th><th>Entropy</th><th>Reasons</th></tr></thead>",
        "<tbody>",
    ]
    for item in items:
        rows.append(
            "<tr>"
            f"<td>{html.escape(item.item_id)}</td>"
            f"<td>{html.escape(item.estimated_label)}</td>"
            f"<td>{item.confidence:.2f}</td>"
            f"<td>{html.escape(item.majority_label)}</td>"
            f"<td>{item.majority_fraction:.2f}</td>"
            f"<td>{item.normalized_entropy:.2f}</td>"
            f"<td>{html.escape(', '.join(item.reasons))}</td>"
            "</tr>"
        )
    rows.extend(["</tbody>", "</table>"])
    return "\n".join(rows)


_CSS = """
:root { color-scheme: light; }
body { margin: 0; font-family: "Iowan Old Style", Palatino, "Palatino Linotype", Georgia, serif;
  background: #f4f6f8; color: #172033; }
header, section, footer { max-width: 1100px; margin: 0 auto; padding: 0 24px; }
header { padding-top: 32px; }
h1 { font-size: 32px; margin: 0 0 8px; letter-spacing: -0.02em; }
h2 { font-size: 20px; margin: 28px 0 8px; }
.eyebrow { text-transform: uppercase; letter-spacing: 0.12em; font-size: 12px;
  font-family: ui-sans-serif, system-ui, sans-serif; color: #5c6b80; margin: 0; }
.source { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; color: #5c6b80; }
.stats { display: flex; flex-wrap: wrap; gap: 12px; padding-top: 8px; }
.stat { background: white; border: 1px solid #d9dee7; border-radius: 8px; min-width: 120px; padding: 12px 14px; }
.stat span { display: block; font-family: ui-sans-serif, system-ui, sans-serif; font-size: 12px; color: #5c6b80; }
.stat strong { font-size: 22px; }
.note, footer p { font-family: ui-sans-serif, system-ui, sans-serif; color: #3e4c63; font-size: 14px;
  line-height: 1.45; }
table { width: 100%; border-collapse: collapse; background: white; font-family: ui-sans-serif, system-ui, sans-serif;
  font-size: 13px; }
th, td { border-bottom: 1px solid #e6ebf2; text-align: left; padding: 8px 10px; vertical-align: top; }
th { color: #5c6b80; font-weight: 600; }
.badge { border-radius: 999px; padding: 2px 8px; font-size: 12px; }
.badge.high { background: #fff1f2; color: #9f1239; }
.badge.medium { background: #fffbeb; color: #92400e; }
.badge.low { background: #eff6ff; color: #1d4ed8; }
.badge.ok { background: #ecfdf3; color: #047857; }
table.confusion { width: auto; margin: 4px 0 12px; }
table.confusion caption { text-align: left; color: #5c6b80; font-size: 12px; margin-bottom: 4px; }
footer { padding-bottom: 40px; }
"""
