## output-auto-fund-2

The block is built: `funding/auto-fund-2/` holds five business days of funding actions for CEO Kevin Kawchak, from 28Sep26 to 02Oct26, each ending in one approval step. The branch `claude/adoring-ramanujan-i3x623` carries 134 commits, one per file or section as the prompt asks, but only the first 55 reached GitHub: from the 29Sep26 packet onward every push was refused with a 403, so the last 79 commits exist only in this session's clone (details under **Push status**).

### What each day answers

| Day | Signal | Sent | Packet |
|:--|:--|:--|:--|
| 1, 28Sep26 | Two unsolicited LinkedIn applicants, no posting | 2 LinkedIn replies, 3 emails | The Unsolicited Applicants |
| 2, 29Sep26 | The SEC San Francisco office taking the request "very seriously" | 5 emails | The External Standard Request |
| 3, 30Sep26 | Several White House communications after the September 12 message | 4 emails, 1 web form | The Structured Team |
| 4, 01Oct26 | An NSF inquiry about a multi-million-dollar pilot | 4 emails, 2 submissions | The Pilot Inquiry |
| 5, 02Oct26 | None; the week's record | 3 earned follow-ups, 2 conditional LinkedIn replies | The Week's Record |

Every day has READMEs with badges and source maps, plus `emails/`, `briefs/`, `forms/`, `investing/`, `diagrams/` and a seven-section `packet/`. Days 1 and 5 also have `linkedin/`. In total: 19 emails, 4 LinkedIn replies, 15 briefs, 7 form packs, 5 capital instructions, 15 figures (three per platform across Mermaid, Graphviz, D2, PlantUML and Diagrams) and 25 tables.

The Rasonque position is stated in every packet: daraxonrasib is FDA approved as Rasonque, and the company's AI papers were developed independently of, and simultaneously with, Revolution Medicines' Phase 1/2 and Phase 3 trials.

### Addresses

The register in `inputs/README.md` holds 36 recipient addresses. Each one was checked against the page its organization publishes. Four candidates were dropped because no publishing page could be found for them.

Every email carries at least three of the 36 addresses across its TO and CC lines. Every body is one paragraph per line, so it pastes into iOS Mail without repair.

Three counterparties publish no usable email address:
- **The two applicants** are answered in the LinkedIn threads they opened.
- **The White House** is reached through its contact form.
- **The brokerage** is reached through its portal.

None of them is emailed.

### Compile record

| Packet | Pages | Errors | Overfull | References | Overleaf zip tested alone |
|:--|:--|:--|:--|:--|:--|
| 28Sep26 | 10 | 0 | 0 | 44 | 0 errors, 0 overfull |
| 29Sep26 | 10 | 0 | 0 | 37 | 0 errors, 0 overfull |
| 30Sep26 | 10 | 0 | 0 | 39 | 0 errors, 0 overfull |
| 01Oct26 | 10 | 0 | 0 | 37 | 0 errors, 0 overfull |
| 02Oct26 | 10 | 0 | 0 | 41 | 0 errors, 0 overfull |

- **Captions:** all 40 are two lines, set -0.60cm below their float.
- **Tables:** all 25 are exactly `\textwidth`, with raggedright columns.
- **References:** every one has a clickable url, plus a doi and a doi url wherever an identifier exists.
- **Page shape:** no page ends with a stranded heading.

### Defects found and fixed

- **CI.** `lint-and-format (3.10)` failed on the pushed head, and 3.11 and 3.12 were cancelled after it.
  - Cause: ruff 0.16 formats Python code blocks inside Markdown, and the Diagrams source in `28Sep26/diagrams/fig-02` used hanging indentation.
  - Fix: the block was reformatted in the audit commit.
  - Verification: with ruff 0.16.9, the version CI installs, `ruff format --check`, `ruff check` and `yamllint` all pass on the whole repository.
- **Day 1 empty page.** The section float barrier flushed Figure 2 onto a page that was two-thirds empty. Moving the figure ahead of Table 3 fixed it and cut the packet from 11 pages to 10.
- **Day 4 compile halt.** A style name inside a TikZ `\foreach` halted the compile. Figure 10 is now drawn one node per line.
- **Day 5 overfull box.** A 0.41 pt overfull box in the reference columns was removed by dropping one uncited entry.
- **Stranded References heading (days 1 to 3).** Room is now reserved before the heading.
- **Tables and captions.**
  - Tables with unevenly wrapping rows were rebalanced.
  - Two figure specifications quoted outdated caption line lengths.
  - Seventeen planned column widths in the sub-prompts did not match the set tables.
  - Nine sentences used gendered pronouns for people whose pronouns are not stated. They now use neutral wording.

### Instructions interpreted rather than followed literally

- **White House communications.** Their text is not in the repository, so they are never quoted or paraphrased. The September 12 message is quoted verbatim.
- **The SEC office.** Only the words "very seriously" are attributed to it. The OpenAI and Anthropic statement is presented as the company's own.
- **The NSF pilot.** The company does not yet meet the pilot's published gate. The answer states the gate and submits the Project Pitch, the first step on the published route. NSF's SBIR program does not fund trials, so the Pitch proposes only the verification work.
- **Captions "at body width".** Captions are two balanced lines centered within the text measure, matching the established style.
- **Dates.** The block is dated September 28 to October 2, and the release is dated the day it was built, September 26, 2026.

### Push status

Every `git push` since the 29Sep26 packet has returned 403 with this message: "Claude doesn't have GitHub access to kevinkawchak/physical-ai-oncology-trials for your organization."

Where things stand:
- [kevinkawchak/physical-ai-oncology-trials#81](https://github.com/kevinkawchak/physical-ai-oncology-trials/pull/81) is still a draft at 55 commits.
- Its last CI run shows the lint failure above, which is fixed only in the local commits.

To unblock it, reconnect GitHub at https://claude.ai/customize/connectors?auth_start=github&auth_start_force=1, or have an org admin install the Claude GitHub App at https://github.com/apps/claude/installations/select_target. Then run `git push -u origin claude/adoring-ramanujan-i3x623` from this session, which will update the PR. After that the PR's title and body still need updating, and it still needs marking ready for review.

### Not claimed

- No office has taken a position.
- No agreement, award or eligibility exists with the SEC, FDA, NSF, NIH, the White House, UC San Diego or any company.
- Neither applicant is named anywhere.
- The perioperative use of Rasonque remains investigational.
- The $36,330 figure is described as projected everywhere.
