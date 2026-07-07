#!/usr/bin/env python3
"""Verify that the field-link post-pass changed no field geometry or pixels."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

try:
    import fitz
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyMuPDF is required for verification.") from exc


def widget_snapshot(doc: fitz.Document):
    rows = []
    for page_number, page in enumerate(doc):
        widget = page.first_widget
        while widget:
            rows.append(
                {
                    "page": page_number,
                    "name": widget.field_name,
                    "type": widget.field_type,
                    "rect": tuple(round(v, 4) for v in widget.rect),
                    "value": widget.field_value,
                    "font": widget.text_font,
                    "fontsize": round(widget.text_fontsize or 0, 4),
                }
            )
            widget = widget.next
    return rows


def render_digest(page: fitz.Page, matrix: fitz.Matrix) -> str:
    pix = page.get_pixmap(matrix=matrix, alpha=False, annots=True)
    return hashlib.sha256(pix.samples).hexdigest()


def normalize_text(text: str) -> str:
    text = text.replace("^^J", " ")
    return re.sub(r"\s+", " ", text).strip()


def field_widgets(doc: fitz.Document):
    out = {}
    for pno, page in enumerate(doc):
        widget = page.first_widget
        while widget:
            if widget.field_name:
                out[widget.field_name] = (pno, fitz.Rect(widget.rect), widget.field_value or "")
            widget = widget.next
    return out


def verify(before_path: Path, after_path: Path, manifest_path: Path, report_path: Path) -> None:
    before = fitz.open(before_path)
    after = fitz.open(after_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    failures = []
    if len(before) != len(after):
        failures.append(f"page count changed: {len(before)} -> {len(after)}")

    before_widgets = widget_snapshot(before)
    after_widgets = widget_snapshot(after)
    if before_widgets != after_widgets:
        failures.append("AcroForm widget names, values, geometry, or font settings changed")

    changed_pages = []
    matrix = fitz.Matrix(1.25, 1.25)
    for pno in range(min(len(before), len(after))):
        if render_digest(before[pno], matrix) != render_digest(after[pno], matrix):
            changed_pages.append(pno + 1)
    if changed_pages:
        failures.append(f"visible rendering changed on pages: {changed_pages}")

    widgets = field_widgets(after)
    links_checked = 0
    missing_link_targets = []
    truncated_fields = []

    for field in manifest["fields"]:
        name = field["field_name"]
        if name not in widgets:
            missing_link_targets.append({"field": name, "reason": "widget missing"})
            continue
        pno, rect, value = widgets[name]
        page = after[pno]
        page_links = page.get_links()
        expected_pairs = []
        for citation in field["citations"]:
            expected_pairs.extend(zip(citation["keys"], citation["labels"]))

        for key, label in dict.fromkeys(expected_pairs):
            label_rects = [fitz.Rect(q.rect) for q in page.search_for(label, quads=True)]
            label_rects = [r for r in label_rects if rect.contains(fitz.Point((r.x0+r.x1)/2, (r.y0+r.y1)/2))]
            matching = []
            for link in page_links:
                if link.get("kind") != fitz.LINK_GOTO:
                    continue
                lrect = fitz.Rect(link["from"])
                if any(not (lrect & lr).is_empty for lr in label_rects):
                    matching.append(link)
            if not label_rects or not matching:
                missing_link_targets.append({"field": name, "key": key, "label": label})
            else:
                links_checked += 1

        # Ensure the visible appearance reaches the end of the initial value.
        # This catches vertical clipping without requiring OCR.
        visible = normalize_text(page.get_textbox(rect))
        value_norm = normalize_text(value)
        tail_words = value_norm.split()[-4:]
        if tail_words:
            tail = " ".join(tail_words)
            if tail not in visible:
                truncated_fields.append({"field": name, "tail": tail, "visible_tail": " ".join(visible.split()[-8:])})

    if missing_link_targets:
        failures.append(f"{len(missing_link_targets)} citation labels lack matching internal links")
    if truncated_fields:
        failures.append(f"{len(truncated_fields)} cited field appearances may be clipped")

    page_count = len(after)
    widget_count = len(after_widgets)
    report = {
        "before": before_path.name,
        "after": after_path.name,
        "pages": page_count,
        "widgets": widget_count,
        "render_changed_pages": changed_pages,
        "citation_labels_checked": links_checked,
        "missing_link_targets": missing_link_targets,
        "possibly_truncated_fields": truncated_fields,
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    before.close()
    after.close()

    if failures:
        raise RuntimeError("; ".join(failures) + f". See {report_path}")
    print(
        f"Verification PASS: {widget_count} fields unchanged, "
        f"{page_count} pages pixel-identical, {links_checked} citation targets checked."
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("before")
    parser.add_argument("after")
    parser.add_argument("manifest")
    parser.add_argument("--report", default="build/field-citation-verification.json")
    args = parser.parse_args()
    verify(Path(args.before), Path(args.after), Path(args.manifest), Path(args.report))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise
