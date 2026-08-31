# Stage 3, sub-prompt 3 - dialect, orthography, and the proof pass

## Goal

Clause K requires the language, tone and dialect used in La Jolla. Clause P
requires that no spelling or grammatical error survives. This sub-prompt states
both as mechanical checks so neither depends on a reading.

## The dialect word list

Every entry is checked with a case-insensitive grep over `sections/*.tex`,
`main.tex` and `movestyle.sty`. Each must return zero.

| Not used | Used instead |
|:--|:--|
| programme | program |
| centre | center |
| organisation, organise, organised | organization, organize, organized |
| authorise, authorised, authorisation | authorize, authorized, authorization |
| prioritise, prioritised | prioritize, prioritized |
| utilise, utilised | use, used |
| summarise, summarised | summarize, summarized |
| analyse, analysed | analyze, analyzed |
| minimise, maximise | minimize, maximize |
| randomisation, randomised | randomization, randomized |
| recognise, recognised | recognize, recognized |
| standardise, standardised | standardize, standardized |
| specialise, specialised | specialize, specialized |
| tumour | tumor |
| colour, behaviour, labour, favour | color, behavior, labor, favor |
| defence, licence (noun), practise (verb) | defense, license, practice |
| metre, litre, fibre | meter, liter, fiber |
| storey | story |
| kerb | curb |
| whilst, amongst | while, among |
| aluminium | aluminum |
| catalogue, dialogue box | catalog, dialog box |
| enrolment | enrollment |
| judgement | judgment |
| ageing | aging |
| towards (in codified text) | toward |

## Punctuation and symbols

| Check | Must return |
|:--|:--|
| Em dash `—` | 0 |
| En dash `–` used as punctuation | 0 |
| Double hyphen `--` outside a comment | 0 |
| Triple hyphen `---` | 0 |
| Literal `SS` where a section symbol belongs | 0 |
| Straight apostrophe inside prose that should be `'` in LaTeX | 0 |
| Two spaces after a period inside a sentence run | 0 |
| A number and its unit split across a line | 0, enforced with `~` |

## Tone

The register is that of a municipal code and a clinical operations manual read
by a funder: declarative, present tense for standing requirements, `shall` for
an obligation in a codified document and `will` or the present tense in the
operations documents. No marketing adjective. No superlative that a reader
cannot check. Where a claim is uncertain, the uncertainty is stated in the same
sentence rather than in a footnote.

## The proof pass

1. Read every section from the compiled PDF, not the source.
2. Check each number against the source file the stage 1 instruction named.
3. Check every cross-reference resolves to the object it names.
4. Check that each of the fifteen documents can be read on its own, which is
   the test that decides whether a definition must be repeated inside a
   document or can be left to the front matter.
5. Check the abbreviation table against the body: every abbreviation used more
   than once appears in the table, and every table row is used in the body.

## Acceptance

Zero on every check above.

## Commit

Folded into the seventeen section commits, then re-run in the error pass.
