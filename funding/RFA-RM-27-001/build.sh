#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
mkdir -p build

python3 - <<'PY'
try:
    import fitz  # noqa: F401
except ImportError as exc:
    raise SystemExit(
        "PyMuPDF is required for clickable citations inside fields. "
        "Install it with: python3 -m pip install PyMuPDF"
    ) from exc
PY

run_pdflatex() {
  local name="$1"
  if ! pdflatex -interaction=nonstopmode -halt-on-error main.tex >"build/${name}.log" 2>&1; then
    tail -n 120 "build/${name}.log" >&2
    return 1
  fi
  echo "LaTeX pass complete: ${name}"
}

# Rebuild citation order and labels deterministically from the source.
rm -f main.aux main.bbl main.blg main.log main.out main.pdf main.toc
rm -f build/latex-*.log build/bibtex.log
python3 scripts/prepare_field_citations.py scan --root . --main main.tex

run_pdflatex latex-1
if command -v bibtex >/dev/null 2>&1; then
  if ! bibtex main >build/bibtex.log 2>&1; then
    cat build/bibtex.log >&2
    exit 1
  fi
elif command -v bibtex8 >/dev/null 2>&1; then
  if ! bibtex8 main >build/bibtex.log 2>&1; then
    cat build/bibtex.log >&2
    exit 1
  fi
else
  echo "Neither bibtex nor bibtex8 was found." >&2
  exit 1
fi
echo "BibTeX pass complete"

# This pass reads the .bbl and writes natbib's resolved author/year labels.
run_pdflatex latex-2
python3 scripts/prepare_field_citations.py labels --root . --main main.tex --aux main.aux

# These passes put the final citation strings in the widgets and settle refs.
run_pdflatex latex-3
run_pdflatex latex-4

cp main.pdf build/main-prelinks.pdf
python3 scripts/link_field_citations.py \
  build/main-prelinks.pdf \
  grant-field-citation-manifest.json \
  --output build/main-linked.pdf \
  --report build/field-citation-link-report.json
mv build/main-linked.pdf main.pdf

python3 scripts/verify_field_citations.py \
  build/main-prelinks.pdf \
  main.pdf \
  grant-field-citation-manifest.json \
  --report build/field-citation-verification.json

cp main.pdf sample_template.pdf
rm -f build/main-prelinks.pdf

echo "Built and verified: $ROOT/main.pdf"
echo "Compiled preview:  $ROOT/sample_template.pdf"
