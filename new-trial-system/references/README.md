# references - source bibliographies

[![Repository](https://img.shields.io/badge/Repository-v4.6.0-800020.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials)
[![Entries](https://img.shields.io/badge/Entries-99%20source%20%2B%2023%20added-A32A3C.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/references)

## Files

| File | Entries | Contents |
|:--|:--|:--|
| [references.bib](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/references/references.bib) | 41 | The author's deposited works, the codified and policy entries, and the clinical literature carried forward from `funding/pdac-funding-applications/final-apply`. |
| [trump-ai-cancer-2025-2026.bib](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/references/trump-ai-cancer-2025-2026.bib) | 58 | Every executive order, agency action, funding change and press record of the 2025 to 2026 Federal AI and cancer program, used by the Introduction and the Conclusions. |

## How they are used

Both files are concatenated verbatim into the `references.bib` of each build
stage, and then extended with the 23 deposits and inputs this paper reads
directly. The merged file carries 122 entries with no duplicate keys.

| Merged into | Path |
|:--|:--|
| Stage 6 | [draft-new-trial/references.bib](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/draft-new-trial/references.bib) |
| Stage 7 | [full-new-trial/references.bib](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/full-new-trial/references.bib) |
| Stage 8 | [final-new-trial/references.bib](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/final-new-trial/references.bib) |

## Convention

Every entry with a DOI carries both a `doi` field and a `url` field pointing at
`https://doi.org/<doi>`. Every entry without a DOI carries a `url` that resolves
to the document itself and never to a search page. The bibliography style is
`unsrturl`, not `unsrt`: `unsrt` reads neither field, so a reference list built
with it prints no clickable target at all.
