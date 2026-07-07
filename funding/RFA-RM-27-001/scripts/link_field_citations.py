#!/usr/bin/env python3
"""Add invisible internal links over citation labels inside AcroForm fields.

PDF text fields cannot natively contain independent hyperlinks. This post-build
step keeps the original widgets and their appearances untouched, then overlays
small, invisible Link annotations on the compiled citation labels. Each link
jumps to the matching hyperref bibliography destination.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "PyMuPDF is required for clickable citations inside PDF form fields. "
        "Install it with: python3 -m pip install PyMuPDF"
    ) from exc


def intersection_fraction(a: fitz.Rect, b: fitz.Rect) -> float:
    inter = a & b
    if inter.is_empty or a.get_area() <= 0:
        return 0.0
    return inter.get_area() / a.get_area()


def widget_index(doc: fitz.Document):
    by_name = {}
    for page_number, page in enumerate(doc):
        widget = page.first_widget
        while widget:
            if widget.field_name:
                by_name[widget.field_name] = (page_number, fitz.Rect(widget.rect), widget.field_value)
            widget = widget.next
    return by_name


def unique_preserve(items):
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def add_links(pdf_path: Path, manifest_path: Path, output_path: Path, report_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    doc = fitz.open(pdf_path)
    names = doc.resolve_names()
    widgets = widget_index(doc)

    expected_by_field: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for field in manifest["fields"]:
        for citation in field["citations"]:
            for key, label in zip(citation["keys"], citation["labels"]):
                expected_by_field[field["field_name"]].append((key, label))

    report = {
        "input_pdf": pdf_path.name,
        "output_pdf": output_path.name,
        "fields_with_citations": len(expected_by_field),
        "links_added": 0,
        "missing_fields": [],
        "missing_destinations": [],
        "unlocated_labels": [],
        "links": [],
    }

    # Inserting an annotation can invalidate MuPDF's cached text view of widget
    # appearances. Locate every rectangle first, then mutate the PDF in a
    # separate pass.
    planned: list[dict] = []
    for field_name, pairs in expected_by_field.items():
        if field_name not in widgets:
            report["missing_fields"].append(field_name)
            continue
        page_number, widget_rect, _ = widgets[field_name]
        page = doc[page_number]
        for key, label in unique_preserve(pairs):
            destination = names.get(f"cite.{key}")
            if not destination:
                report["missing_destinations"].append({"field": field_name, "key": key})
                continue

            quads = page.search_for(label, quads=True)
            matching_quads = []
            for quad in quads:
                rect = fitz.Rect(quad.rect)
                center = fitz.Point((rect.x0 + rect.x1) / 2, (rect.y0 + rect.y1) / 2)
                if widget_rect.contains(center) or intersection_fraction(rect, widget_rect) >= 0.75:
                    matching_quads.append(quad)

            if not matching_quads:
                report["unlocated_labels"].append(
                    {"field": field_name, "key": key, "label": label, "page": page_number + 1}
                )
                continue

            dest_point = fitz.Point(destination["to"])
            for quad in matching_quads:
                rect = fitz.Rect(quad.rect)
                rect = fitz.Rect(rect.x0 - 0.35, rect.y0 - 0.2, rect.x1 + 0.35, rect.y1 + 0.2)
                rect &= widget_rect
                planned.append(
                    {
                        "page_number": page_number,
                        "rect": rect,
                        "destination_page": int(destination["page"]),
                        "destination_point": dest_point,
                        "zoom": float(destination.get("zoom", 0.0)),
                        "field": field_name,
                        "key": key,
                        "label": label,
                    }
                )

    if report["missing_fields"] or report["missing_destinations"] or report["unlocated_labels"]:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        doc.close()
        raise RuntimeError("citation-link pass was incomplete; see " + str(report_path))

    for item in planned:
        page = doc[item["page_number"]]
        page.insert_link(
            {
                "kind": fitz.LINK_GOTO,
                "from": item["rect"],
                "page": item["destination_page"],
                "to": item["destination_point"],
                "zoom": item["zoom"],
            }
        )
        report["links_added"] += 1
        report["links"].append(
            {
                "field": item["field"],
                "key": item["key"],
                "label": item["label"],
                "source_page": item["page_number"] + 1,
                "rect": [round(v, 3) for v in item["rect"]],
                "destination_page": item["destination_page"] + 1,
                "destination": [
                    round(item["destination_point"].x, 3),
                    round(item["destination_point"].y, 3),
                ],
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if temp_path.exists():
        temp_path.unlink()
    doc.save(temp_path, garbage=4, deflate=True)
    doc.close()
    os.replace(temp_path, output_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Added {report['links_added']} clickable citation link rectangles in {len(expected_by_field)} fields.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf")
    parser.add_argument("manifest")
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", default="build/field-citation-link-report.json")
    args = parser.parse_args()
    add_links(Path(args.pdf), Path(args.manifest), Path(args.output), Path(args.report))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise
