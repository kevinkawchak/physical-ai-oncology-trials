# Simplified Clinical Trial Funding Application Template

Open **[START_HERE.md](START_HERE.md)** for the direct editing index.

This is a complete LaTeX project for a fillable, funder-adaptable clinical-trial application. The majority of the original sections remain intact, but field editing is now more direct:

- Every section is linked from the start page.
- Text is entered in the same field block that controls its layout.
- Every field shows its font size beside the text, usually `font=10pt` or `font=9pt`.
- Change that value locally, or use `font=auto` for PDF auto-fit behavior.
- Shared applicant and opportunity details remain in two short configuration files.

## Main files

- [Start page and section links](START_HERE.md)
- [Browser-friendly file navigator](START_HERE.html)
- [Master document and section order](main.tex)
- [Applicant defaults](config/applicant.tex)
- [Opportunity defaults](config/opportunity.tex)
- [Field and page styling](clinicaltrialgrant.sty)
- [Bibliography](references.bib)
- [Compiled preview](sample_template.pdf)

## Build

```bash
python3 -m pip install -r requirements.txt
./build.sh
```

The script runs LaTeX/BibTeX, resolves field-safe author-year labels, adds the internal citation links, and verifies that form geometry and visible rendering did not change during the link pass.

The project uses standard TeX Live packages including `hyperref`, `geometry`, `fancyhdr`, `tabularx`, `natbib`, `xcolor`, and `xparse`. The verified `build.sh` workflow also uses Python 3 with PyMuPDF (`python3 -m pip install -r requirements.txt`) to add invisible internal links over citations that appear inside fillable fields.

## Field syntax

```latex
\GrantTextArea[
  id=field-name, font=10pt, height=2in, maxchars=2000
]{Visible heading}{%
  Your text goes here.
}
```

The generated PDF remains fillable. Text entered in the LaTeX source becomes the initial field value, and the field can still be edited in a compatible PDF viewer.

### Citations inside fields

Use ordinary natbib syntax in a field value, for example:

```latex
\GrantTextArea[id=example,font=10pt,height=1.5in]{Example}{%
  This statement is supported by prior work \citep{reference_key}.
}
```

Run `./build.sh`, not a single direct `pdflatex` command. The build resolves the natbib author-year text, keeps every form widget's size and appearance unchanged, and overlays a tight invisible link on each compiled citation label. Clicking a label jumps to its matching bibliography entry. The links correspond to the initial compiled field text; rebuild after changing citation text or source content.

## Important use notice

This package is a drafting aid, not an official application package. It does not replace Grants.gov Workspace, NIH ASSIST, SF424 forms, a sponsor portal, IRB review, FDA submissions, institutional authorization, legal review, or the live solicitation. Verify every final application against current funder instructions.
