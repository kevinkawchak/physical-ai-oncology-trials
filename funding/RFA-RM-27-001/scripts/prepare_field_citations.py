#!/usr/bin/env python3
"""Prepare field-safe natbib citation strings and a link manifest.

This script has two phases:
  scan   - recursively scan main.tex, generate \nocite entries and placeholder
           field citation strings so the first LaTeX pass is safe.
  labels - parse natbib's \bibcite records from main.aux, generate final field
           citation strings and a JSON manifest used by the PDF link pass.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

FIELD_ARITY = {
    "GrantLineField": 2,
    "GrantInlineField": 2,
    "GrantTextArea": 2,
    "GrantPageArea": 2,
    "GrantTableField": 1,
}


@dataclass
class CitationGroup:
    raw: str
    keys: list[str]


@dataclass
class FieldRecord:
    field_name: str
    source: str
    macro: str
    citations: list[CitationGroup]


def strip_comments(text: str) -> str:
    out: list[str] = []
    for line in text.splitlines(keepends=True):
        cut = None
        for i, ch in enumerate(line):
            if ch != "%":
                continue
            backslashes = 0
            j = i - 1
            while j >= 0 and line[j] == "\\":
                backslashes += 1
                j -= 1
            if backslashes % 2 == 0:
                cut = i
                break
        if cut is None:
            out.append(line)
        else:
            suffix = "\n" if line.endswith("\n") else ""
            out.append(line[:cut] + suffix)
    return "".join(out)


def skip_ws(text: str, pos: int) -> int:
    while pos < len(text) and text[pos].isspace():
        pos += 1
    return pos


def parse_balanced(text: str, pos: int, opener: str, closer: str) -> tuple[str, int]:
    if pos >= len(text) or text[pos] != opener:
        raise ValueError(f"expected {opener!r} at offset {pos}")
    depth = 1
    i = pos + 1
    start = i
    while i < len(text):
        ch = text[i]
        if ch == "\\":
            i += 2
            continue
        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start:i], i + 1
        i += 1
    raise ValueError(f"unbalanced {opener}{closer} starting at offset {pos}")


def normalize_group_raw(raw: str) -> str:
    return re.sub(r"\s+", " ", raw.strip())


def parse_citep_groups(text: str) -> list[CitationGroup]:
    groups: list[CitationGroup] = []
    for match in re.finditer(r"\\citep\b", text):
        pos = skip_ws(text, match.end())
        # natbib permits up to two optional arguments. They do not affect the
        # field label generated here, but skipping them makes scanning robust.
        optional_count = 0
        while pos < len(text) and text[pos] == "[" and optional_count < 2:
            _, pos = parse_balanced(text, pos, "[", "]")
            pos = skip_ws(text, pos)
            optional_count += 1
        if pos >= len(text) or text[pos] != "{":
            continue
        raw, _ = parse_balanced(text, pos, "{", "}")
        normalized = normalize_group_raw(raw)
        keys = [key.strip() for key in normalized.split(",") if key.strip()]
        if keys:
            groups.append(CitationGroup(raw=normalized, keys=keys))
    return groups


def parse_field_records(text: str, source: str) -> list[FieldRecord]:
    records: list[FieldRecord] = []
    pattern = re.compile(r"\\(" + "|".join(map(re.escape, FIELD_ARITY)) + r")\b")
    for match in pattern.finditer(text):
        macro = match.group(1)
        pos = skip_ws(text, match.end())
        options = ""
        try:
            if pos < len(text) and text[pos] == "[":
                options, pos = parse_balanced(text, pos, "[", "]")
                pos = skip_ws(text, pos)
            args: list[str] = []
            for _ in range(FIELD_ARITY[macro]):
                if pos >= len(text) or text[pos] != "{":
                    raise ValueError("missing required argument")
                arg, pos = parse_balanced(text, pos, "{", "}")
                args.append(arg)
                pos = skip_ws(text, pos)
        except ValueError as exc:
            print(f"warning: could not parse {macro} in {source}: {exc}", file=sys.stderr)
            continue

        id_match = re.search(r"(?:^|,)\s*id\s*=\s*([^,\]]+)", options)
        if not id_match:
            continue
        field_name = id_match.group(1).strip()
        value = args[-1]
        citations = parse_citep_groups(value)
        if citations:
            records.append(
                FieldRecord(
                    field_name=field_name,
                    source=source,
                    macro=macro,
                    citations=citations,
                )
            )
    return records


def ordered_tex_files(root: Path, entry: Path) -> list[Path]:
    ordered: list[Path] = []
    active: set[Path] = set()

    def visit(path: Path) -> None:
        path = path.resolve()
        if path in active:
            raise RuntimeError(f"recursive \\input loop involving {path}")
        if not path.exists():
            raise FileNotFoundError(path)
        active.add(path)
        ordered.append(path)
        text = strip_comments(path.read_text(encoding="utf-8"))
        for match in re.finditer(r"\\(?:input|include)\s*\{([^}]+)\}", text):
            target = match.group(1).strip()
            candidate = (path.parent / target)
            if candidate.suffix == "":
                candidate = candidate.with_suffix(".tex")
            if candidate.exists():
                visit(candidate)
        active.remove(path)

    visit((root / entry).resolve())
    return ordered


def unique_in_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def macro_name(raw: str) -> str:
    # TeX tokenization collapses runs of source whitespace to one space token.
    return normalize_group_raw(raw)


def tex_detokenize(text: str) -> str:
    # \detokenize cannot contain an unmatched brace. Citation labels generated
    # from natbib are plain text, but guard braces defensively.
    return text.replace("{", "(").replace("}", ")")


def write_mapping(path: Path, groups: list[CitationGroup], label_for_key: dict[str, str] | None) -> None:
    lines = [
        "% Generated by scripts/prepare_field_citations.py; do not edit.",
    ]
    seen_names: set[str] = set()
    for group in groups:
        name = macro_name(group.raw)
        if name in seen_names:
            continue
        seen_names.add(name)
        if label_for_key is None:
            rendered = "[citation pending]"
        else:
            rendered_parts = []
            missing = []
            for key in group.keys:
                label = label_for_key.get(key)
                if label is None:
                    missing.append(key)
                else:
                    rendered_parts.append(label)
            if missing:
                raise RuntimeError(f"missing natbib labels for: {', '.join(missing)}")
            rendered = "[" + "; ".join(rendered_parts) + "]"
        rendered = tex_detokenize(rendered)
        lines.append(
            "\\expandafter\\def\\csname GrantFieldCite@"
            + name
            + "\\endcsname{\\detokenize{"
            + rendered
            + "}}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_braced_argument(text: str, pos: int) -> tuple[str, int]:
    pos = skip_ws(text, pos)
    return parse_balanced(text, pos, "{", "}")


def top_level_brace_groups(text: str) -> list[str]:
    groups: list[str] = []
    pos = 0
    while True:
        pos = skip_ws(text, pos)
        if pos >= len(text):
            break
        if text[pos] != "{":
            pos += 1
            continue
        group, pos = parse_balanced(text, pos, "{", "}")
        groups.append(group)
    return groups


def latex_plain(text: str) -> str:
    text = re.sub(r"\\natexlab\s*\{([^{}]*)\}", r"\1", text)
    replacements = {
        r"\\&": "&",
        r"\\%": "%",
        r"\\_": "_",
        r"\\#": "#",
        r"\\textasciitilde": "~",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    # Drop simple formatting commands while keeping their contents.
    text = re.sub(r"\\(?:textit|textbf|emph|mathrm|operatorname)\s*\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\[A-Za-z@]+", "", text)
    text = text.replace("{", "").replace("}", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_bibcite_labels(aux_path: Path) -> dict[str, str]:
    text = aux_path.read_text(encoding="utf-8", errors="replace")
    labels: dict[str, str] = {}
    pos = 0
    needle = r"\bibcite"
    while True:
        idx = text.find(needle, pos)
        if idx < 0:
            break
        cur = idx + len(needle)
        try:
            key, cur = parse_braced_argument(text, cur)
            payload, cur = parse_braced_argument(text, cur)
        except ValueError:
            pos = idx + len(needle)
            continue
        parts = top_level_brace_groups(payload)
        if len(parts) >= 3:
            year = latex_plain(parts[1])
            author = latex_plain(parts[2])
            labels[key.strip()] = f"{author}, {year}"
        pos = cur
    return labels


def collect(root: Path, main: str) -> tuple[list[Path], list[CitationGroup], list[FieldRecord]]:
    files = ordered_tex_files(root, Path(main))
    all_groups: list[CitationGroup] = []
    fields: list[FieldRecord] = []
    for path in files:
        rel = path.relative_to(root.resolve()).as_posix()
        text = strip_comments(path.read_text(encoding="utf-8"))
        all_groups.extend(parse_citep_groups(text))
        fields.extend(parse_field_records(text, rel))
    return files, all_groups, fields


def phase_scan(root: Path, main: str) -> None:
    _, groups, fields = collect(root, main)
    keys = unique_in_order(key for group in groups for key in group.keys)
    (root / "grant-field-citations.tex").write_text(
        "% Generated by scripts/prepare_field_citations.py; do not edit.\n"
        + ("\\nocite{" + ",".join(keys) + "}\n" if keys else ""),
        encoding="utf-8",
    )
    write_mapping(root / "grant-field-cite-labels.tex", groups, None)
    scan = {
        "main": main,
        "citation_keys": keys,
        "citation_groups": [asdict(group) for group in groups],
        "fields": [
            {
                "field_name": field.field_name,
                "source": field.source,
                "macro": field.macro,
                "citations": [asdict(c) for c in field.citations],
            }
            for field in fields
        ],
    }
    (root / "grant-field-citation-scan.json").write_text(
        json.dumps(scan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Scanned {len(groups)} citation groups ({len(keys)} unique keys) in {len(fields)} cited fields.")


def phase_labels(root: Path, main: str, aux: str) -> None:
    _, groups, fields = collect(root, main)
    labels = parse_bibcite_labels(root / aux)
    keys = unique_in_order(key for group in groups for key in group.keys)
    missing = [key for key in keys if key not in labels]
    if missing:
        raise RuntimeError(
            "natbib labels are unavailable for: " + ", ".join(missing) + ". Run BibTeX and one LaTeX pass first."
        )
    write_mapping(root / "grant-field-cite-labels.tex", groups, labels)
    manifest = {
        "main": main,
        "labels": labels,
        "fields": [
            {
                "field_name": field.field_name,
                "source": field.source,
                "macro": field.macro,
                "citations": [
                    {
                        "raw": citation.raw,
                        "keys": citation.keys,
                        "labels": [labels[key] for key in citation.keys],
                        "rendered": "[" + "; ".join(labels[key] for key in citation.keys) + "]",
                    }
                    for citation in field.citations
                ],
            }
            for field in fields
        ],
    }
    (root / "grant-field-citation-manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Generated final field labels for {len(keys)} references in {len(fields)} cited fields.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("scan", "labels"))
    parser.add_argument("--root", default=".")
    parser.add_argument("--main", default="main.tex")
    parser.add_argument("--aux", default="main.aux")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    if args.phase == "scan":
        phase_scan(root, args.main)
    else:
        phase_labels(root, args.main, args.aux)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise
