# Start Here - Simplified Editing

This project keeps the original grant structure, but you no longer need to study the style file to enter text or change a field's font size.

## 1. Edit the two shared setup files

- [Applicant information](config/applicant.tex)
- [Funding opportunity information](config/opportunity.tex)

## 2. Open the section you want to edit

- [Cover and transmittal](sections/00-cover-and-transmittal.tex)
- [Opportunity and applicant organization](sections/01-opportunity-and-organization.tex)
- [Project summary and public narrative](sections/02-project-summary.tex)
- [Research strategy / five-page essay](sections/03-research-strategy.tex)
- [Clinical trial synopsis](sections/04-clinical-trial-synopsis.tex)
- [Human subjects, ethics, inclusion, and regulatory strategy](sections/05-human-subjects-and-regulatory.tex)
- [Statistics, safety, recruitment, and operations](sections/06-statistics-safety-and-operations.tex)
- [Data management, sharing, repositories, and cybersecurity](sections/07-data-management-and-sharing.tex)
- [Facilities and other resources](sections/08-facilities-and-resources.tex)
- [Budget, milestones, and sustainability](sections/09-budget-milestones-and-sustainability.tex)
- [Investigator profile and current support](sections/10-investigator-profile.tex)
- [Assurances and attachment checklist](sections/11-assurances-and-attachments.tex)
- [Template notes and source record](sections/99-template-notes.tex)

The links above open the actual source files in editors that support relative Markdown links. A browser-friendly link page is also included as [START_HERE.html](START_HERE.html).

## 3. Type directly in a field block

A large field looks like this:

```latex
\GrantTextArea[
  id=project-abstract, font=10pt, height=3.00in, maxchars=3200
]{Project Summary / Abstract}{%
  Type or paste your text here.
}
```

The text is entered inside the final braces. The font setting is in the same block:

- Most large and one-line fields start at `font=10pt`; compact table fields start at `font=9pt`.
- Change the number in that same field block whenever needed.
- `font=auto` is still available when you prefer PDF auto-fit behavior.

A one-line field is equally direct:

```latex
\GrantLineField[id=project-title,font=10pt,maxchars=220]
  {Project title}
  {Type the project title here}
```

You normally do not need to edit [clinicaltrialgrant.sty](clinicaltrialgrant.sty).

## 4. Choose included sections in one place

Open [main.tex](main.tex). The section files are listed in document order. Comment out an `\input{...}` line to omit that section. Change `\TemplateAppendixtrue` to `\TemplateAppendixfalse` to remove the notes and bibliography from the final PDF.

## 5. Compile

Install the one post-processing dependency, then run the verified build:

```bash
python3 -m pip install -r requirements.txt
./build.sh
```

A plain `pdflatex`/BibTeX build will render field citations as text, but `build.sh` is required to add the clickable citation overlays and run the field-geometry and pixel-identity checks.
