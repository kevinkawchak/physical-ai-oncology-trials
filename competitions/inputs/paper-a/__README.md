# README — CodeClash Paper Chunked Files
## For Claude Code Opus 4.7 (1M Context) Processing

**Paper:** "CodeClash: Benchmarking Goal-Oriented Software Engineering"
**arXiv ID:** arXiv:2511.00839v1
**Authors:** John Yang*, Kilian Lieret*, Joyce Yang, Carlos E. Jimenez, Ofir Press, Ludwig Schmidt, Diyi Yang
**Affiliations:** Stanford University, Princeton University, Cornell University
**Open Source:** https://github.com/CodeClash-ai/CodeClash | leaderboard/viewer: https://codeclash.ai

---

## Purpose of This Document

This README is provided to guide Claude Code Opus 4.7 in processing ten chunked Markdown files derived from the CodeClash paper (arXiv:2511.00839v1). Each chunk preserves the original paper text word-for-word without abbreviation. This document describes: (1) the content of each chunk, (2) how the chunks relate to each other, and (3) cross-file dependencies critical for understanding the paper's architecture for the purpose of producing a new physical AI oncology trial paper.

---

## Overview of Chunk Files

| File | Section(s) Covered | Source Files |
|---|---|---|
| chunk_01_abstract_introduction.md | Abstract, Section 1 Introduction | sections/abstract.tex, sections/introduction.tex |
| chunk_02_codeclash_formulation.md | Section 2: CodeClash (Formulation, Technical Details, Features) | sections/codeclash.tex |
| chunk_03_experiments.md | Section 3: Experiments | sections/experiments.tex |
| chunk_04_results.md | Section 4: Results + Ablations | sections/results.tex |
| chunk_05_analysis.md | Section 5: Analysis (Competitive Dynamics, Strategic Reasoning Limits) | sections/analysis.tex |
| chunk_06_related_work_conclusion_acknowledgments.md | Section 6: Related Works, Section 7: Discussion/Conclusion, Acknowledgments | sections/related_work.tex, sections/conclusion.tex, sections/acknowledgments.tex |
| chunk_07_tables.md | All 5 Tables (Main Results, Arena List, TrueSkill, Rank Stability, Elo Uncertainties) | tables/*.tex |
| chunk_08_appendix_infrastructure_arenas.md | Appendix A: Infrastructure, Appendix B: Arenas (all 7 arena cards) | appendix/infrastructure.tex, appendix/arenas.tex, appendix/arena_cards/*.tex |
| chunk_09_appendix_evaluation_results_prompts.md | Appendix C: Evaluation, Appendix D: Extended Results, LM Judge Prompts | appendix/evaluation.tex, appendix/results.tex, appendix/prompts/*.tex |
| chunk_10_bibtex_references.md | All BibTeX entries (full reference list) | colm2025_conference.bib |

---

## Detailed Description of Each Chunk File

---

### chunk_01_abstract_introduction.md
**Content:** The Abstract and Section 1 (Introduction) of the paper.

**Key information:**
- Introduces CodeClash as a benchmark for goal-oriented software engineering where LMs compete in multi-round coding tournaments.
- States the core experimental scope: 1680 tournaments, 25,200 rounds, 8 LMs, 6 arenas.
- Identifies the central finding: top models (Claude Sonnet 4.5) fail to win a single round against expert human-written bot (gigachad).
- Introduces Figure 1, which describes the two-phase round structure (edit phase → competition phase → log copying).
- States that models hallucinate reasons for failure and do not validate code changes.
- Announces open-source release of CodeClash.

**Correlates with:**
- chunk_02: Expands on the benchmark structure introduced here.
- chunk_03: Expands on "8 frontier LMs" and 1680 tournaments.
- chunk_04: Provides the main results referenced in the introduction.
- chunk_05: Provides the detailed analysis of model limitations described here (hallucination, cascade failures).
- chunk_07: Tables 1 and 2 directly support claims made here.

---

### chunk_02_codeclash_formulation.md
**Content:** Section 2 (CodeClash), covering three subsections: Formulation (2.1), Technical Details (2.2), and Features (2.3).

**Key information:**
- Formalizes the benchmark: players = LMs + ACI scaffold; code arena = competition platform with measurable outcomes.
- Defines three design principles: codebase-as-memory, log-based feedback, strategic opacity.
- Describes `mini-SWE-agent` (bash-only ACI) as the player scaffold.
- Defines 5 distinctive properties: open-ended objectives, diverse arenas, adversarial adaptation, self-crafted memory, self-directed improvement.
- Establishes that models cannot see opponents' codebases (default) — transparency is an ablation.
- Confirms 6 code arenas in initial release.

**Correlates with:**
- chunk_01: Introduces the concepts that this chunk formalizes.
- chunk_03: Refers back to this chunk's agent system description for experiment setup.
- chunk_04: The ablation on transparent codebases is referenced in Section 4.1.
- chunk_07: Table 2 (arena list) directly enumerates the arenas introduced here.
- chunk_08: Appendix A and B expand on the mini-SWE-agent and each arena respectively.
- chunk_09: Appendix C provides full prompt configurations for mini-SWE-agent.

---

### chunk_03_experiments.md
**Content:** Section 3 (Experiments) — models evaluated, agent system, round counts, win rate definitions, and Elo metric description.

**Key information:**
- 8 models: Claude Sonnet 4.5, Claude Sonnet 4, GPT-5, GPT-5-mini, o3, Gemini 2.5 Pro, Qwen3-Coder, Grok Code Fast 1.
- Uses `mini-SWE-agent` with bash-only interaction; 30 turn limit per round.
- Tournament structure: C(8,2) × 6 arenas × 10 tournaments × 15 rounds = 25,200 rounds total.
- Tournament win = most rounds won (or last win in tie).
- Elo via maximum likelihood fit to win rates (not sequential K-factor updates); base R=1200, slope β=400.
- 98%+ pairwise order agreement on rank stability.

**Correlates with:**
- chunk_01: States these are the 8 models, 6 arenas, 1680 tournaments referenced in the abstract.
- chunk_04: The results that emerge from this experimental design.
- chunk_07: Table 5 (Elo with uncertainties) and Table 4 (rank stability metrics) support claims here.
- chunk_09: Appendix C.1–C.3 provides full technical specification of agent config, tournament config, and Elo math.

---

### chunk_04_results.md
**Content:** Section 4 (Results) including main leaderboard findings and three ablations.

**Key information:**
- **Main result:** Claude Sonnet 4.5 (Elo 1389) > o3 (1343) > GPT-5 (1360) > Claude Sonnet 4 (1223) > GPT-5 mini (1200) > Gemini 2.5 Pro (1125) > Grok Code Fast (1004) > Qwen3 Coder (952).
- No model dominates all arenas; Claude Sonnet 4.5 ranks only 4th in Poker.
- **Ablation 1 (Human vs AI):** Claude Sonnet 4.5 wins 0 of 150 rounds vs static `gigachad` bot (RobotRumble).
- **Ablation 2 (Transparent codebases):** With opponent codebase visible, GPT-5 gains +7.8%, Claude 4.5 loses -1.8%, Gemini loses -5.5%; frequent inspection doesn't guarantee wins.
- **Ablation 3 (Multi-player):** 6-player Core War shows similar rankings but far more volatility; TrueSkill used instead of Elo.
- References Figure 2 (win rate heatmap) and Figure 3 (per-round win rate line chart).

**Correlates with:**
- chunk_07: Table 1 (Main Results Elo per arena), Table 2 (TrueSkill) are the primary quantitative companions.
- chunk_03: This is the result of the experimental design specified there.
- chunk_05: Analysis section explains *why* these results occur (log interpretation failures, hallucination, etc.).
- chunk_09: Appendix D.2 provides additional ablation data (multi-player volatility charts, opponent access timing).

---

### chunk_05_analysis.md
**Content:** Section 5 (Analysis), covering two subsections: Competitive Dynamics (5.1) and Strategic Reasoning Limitations (5.2).

**Key information:**
- **Edit behavior diversity:** o3 edits ~2 files, ~51 lines/round; Claude Sonnet 4.5 edits ~400+ lines; Gemini 2.5 Pro averages 105 words per thought; no correlation found between these behaviors and win rate.
- **Comeback failure:** After 1 loss, Claude Sonnet 4.5 win probability drops from 71% to <33%. After 5 losses, all models fall below 15%.
- **Solution diversity:** Models' solutions become more dissimilar each round; o3 starts at 0.63 similarity, GPT-5 at 0.41.
- **Codebase mess:** File creation scales linearly with rounds; Claude Sonnet 4.5 creates 30+ files per tournament; filename redundancy 59% for Qwen3-Coder, 35% for Claude Sonnet models; throwaway files high for Claude (18/tournament) and GPT-5 (15).
- **Log analysis failure:** Most models fail to extract insights from logs; o3 has ungrounded edits in ~80% of rounds.
- **Hallucination:** Claude Sonnet 4.5 makes uncorroborated loss claims in >17% of rounds; up to 46% in BattleSnake.
- **Validation failure:** Only Claude Sonnet 4.5 (56%) and GPT-5 (50%) validate in majority of rounds; Gemini/o3 validate only 1 in 5 rounds.
- **Bash error rates:** 85%+ of actions succeed; recovery from errors is >80% on very next step.

**Correlates with:**
- chunk_04: These analyses explain the results seen in chunk_04.
- chunk_07: Tables 1 and 5 provide the numeric context for model rankings discussed here.
- chunk_09: Appendix D.1 (interaction trend charts), D.3 (LM-as-judge methodology), D.4 (codebase organization plots) expand on each subsection of this chunk.

---

### chunk_06_related_work_conclusion_acknowledgments.md
**Content:** Section 6 (Related Works), Section 7 (Discussion/Conclusion), and Acknowledgments.

**Key information:**
- **Related work coverage:** SWE-bench and derivatives (repository-level benchmarks); performance optimization benchmarks (Mercury, KernelBench, GSO); game-playing evaluations (BALROG, GameArena, Atari-GPT); self-improving agents (Huxley-Gödel Machine, Darwin-Gödel Machine).
- **Positioning:** CodeClash is unique in combining interactive coding + competitive gaming; unlike optimization benchmarks, arenas have diverse flexible win conditions; unlike pure game-playing benchmarks, LMs write code rather than directly playing.
- **Limitations:** Arenas are small/self-contained; bash-only scaffold; text-only logs (no VLM); self-play/RL potential unexplored.
- **Future directions:** Larger arenas (cybersecurity, healthcare, city planning); tool-based scaffolds; multimodal feedback; pre-training/post-training on CodeClash traces.
- **Acknowledgments:** Laude Institute, Andreessen Horowitz, Open Philanthropy (funding); PLI Princeton (API credits); bitbop.io (compute).

**Correlates with:**
- chunk_01: The introduction's problem statement is bookended by this conclusion.
- chunk_03: The model list references citations in this section (model cards for GPT-5, o3, Claude Sonnet 4/4.5, Gemini 2.5 Pro, Qwen3-Coder, Grok Code Fast 1).
- chunk_10: All citations in this section have full BibTeX entries in chunk_10.

---

### chunk_07_tables.md
**Content:** All five tables from the paper, rendered as Markdown tables with captions and LaTeX labels.

**Tables included:**
- **Table 1 (tab:main_results):** Elo ratings per model per arena (8 models × 6 arenas + overall).
- **Table 2 (tab:list_arenas):** Arena overview (name, description, player count, language).
- **Table 3 (tab:trueskill):** TrueSkill μ scores for 6-player Core War (6 models).
- **Table 4 (tab:rank_stability_bootstrapping):** Rank stability metrics (Kendall's τ, Spearman's ρ, Footrule, Top-1 consistency, Pairwise order agreement) for parametric and nonparametric bootstrapping.
- **Table 5 (tab:elo_ratings_uncertainties):** Full Elo ratings with ±uncertainties per model per arena.

**Correlates with:**
- chunk_03: Table 4 directly supports the "98% pairwise order agreement" claim in the Experiments section.
- chunk_04: Table 1 and Table 3 are the primary outputs of the Results section.
- chunk_05: All tables provide numerical context for analysis claims.
- chunk_09: Tables 4 and 5 are formally embedded within Appendix C.3 (Evaluation Metrics) content, which is expanded in chunk_09.

---

### chunk_08_appendix_infrastructure_arenas.md
**Content:** Appendix A (Infrastructure, labeled appx:infra) and Appendix B (Arenas, labeled appx:arenas).

**Infrastructure section includes:**
- How models edit codebases: three design principles for LM-codebase interaction (execution feedback, interactivity, bash-only).
- Docker containerization for reproducibility; agent containers vs arena containers.
- Starter codebase assets: docs/, arena executable, working submission.
- 1000 simulations per round for winner determination.
- Log copying from arena container to agent codebase (`logs/` folder).
- Invalid submission handling decision tree.
- Positional advantage detection and random player shuffle fix.
- Log/trajectory viewer tool.
- mini-SWE-agent + Claude 4 Opus achieves 67.6% on SWE-bench Verified (bash-only leaderboard).

**Arenas section covers all 7 arenas with full arena cards:**
- B.1 MIT Battlecode 2025 (Python; paint 70% of map; turn-based RTS; log: turn-by-turn narrative).
- B.2 Battlesnake (Python; last snake standing on 11×11 grid; log: JSONL per-turn snapshots).
- B.3 Core War (Redcode; last warrior in shared memory; log: high-level win/loss/process summary).
- B.4 Halite I (C/C++/OCaml/Rust; territory control on wrapped grid; log: sequential text per turn).
- B.5 Poker / Husky Hold'em Bench (Python; No-Limit Texas Hold'em; log: betting round action sequences in JSON).
- B.6 RoboCode (Java; tank combat with survival/damage scoring; log: cumulative score summary table).
- B.7 RobotRumble (JavaScript/Python; 100-turn robot grid battle; log: ASCII grid sequence).

**Each arena card contains:**
- Game rules and objective.
- System prompt description (verbatim text used to brief LMs).
- Effective strategies.
- Initial codebase assets.
- Arena configurations.
- Winner determination logic.
- Log format description.
- Example log excerpt.

**Correlates with:**
- chunk_02: Section 2.2 (Technical Details) references infrastructure decisions in this appendix.
- chunk_03: Arena-specific constraints and codebase structures are relevant to understanding model performance differences.
- chunk_04: Human vs AI ablation (gigachad) uses RobotRumble (B.7); transparent codebase ablation uses Core War (B.3).
- chunk_05: Analysis of codebase mess and log interpretation failures depends on understanding each arena's file and log structure.
- chunk_09: Appendix C.1 (mini-SWE-agent config) references the infrastructure design described here; arena prompts reference each arena card.

---

### chunk_09_appendix_evaluation_results_prompts.md
**Content:** Appendix C (Evaluation) and Appendix D (Extended Results), plus three verbatim LM judge system prompts.

**Appendix C includes:**
- C.1: mini-SWE-agent configuration: 30-turn limit, $1 cost limit, ReAct prompt format, full system prompt verbatim, arena description prompt verbatim, command execution rules verbatim, format error template verbatim.
- C.2: Tournament configuration: YAML schema with tournament/game/player fields; sims_per_round; parameter calculation (M=9, A=6, T=10, P=2, R=15).
- C.3: Evaluation metrics definitions (round win, tournament win, win rate, Elo/Bradley-Terry MLE formulation, Hessian-based covariance matrix for uncertainties, gauge fixing, parametric and non-parametric bootstrapping procedures).

**Appendix D includes:**
- D.1: Interaction trends (files edited, lines changed, README changes, submission file changes, steps per round, thought length CDFs and line charts, errant action rates by model/arena, recovery speed from errors).
- D.2: Additional ablations (multi-player Core War volatility; transparent codebase opponent access frequency per model per round range).
- D.3: LM-as-judge methodology for groundedness/validation, hallucination, and action categorization studies — includes all Python structured output schema classes.
- D.4: Additional analyses (code evolution heatmaps at round 1 vs round 15; codebase organization scatter plot with file reuse ratio vs root level clutter; filename redundancy over rounds; CDF of total files created per tournament; bloated codebase example screenshot reference).
- Future arenas discussion (cybersecurity, healthcare/EHR, city planning).

**Three verbatim LM judge system prompts:**
- Groundedness and validation study prompt: Defines 8 boolean/categorical questions (Q1–Q8) with precise definitions of "main player file," "final edits," evidence standards.
- Hallucination study prompt: Defines "incidents" with 6 conditions; specifies claim_categories and source_categories for structured output; provides positive and negative examples.
- Action categorization prompt: Defines hierarchical action taxonomy (search, navigate, read.x.y, write.x.y, execute.x.y, submit, other) with subsubcategories; priority rules (execution > writing > reading); base_action definition.

**Correlates with:**
- chunk_02: The system prompt verbatim in C.1 is what models receive in place of the brief description referenced in Section 2.3.
- chunk_03: C.2 and C.3 are the formal mathematical expansion of the win rate and Elo descriptions in Section 3.
- chunk_04: D.2 expands ablation data from Section 4.1.
- chunk_05: D.1, D.3, D.4 provide figures/charts that support all claims in Section 5. The LM judge prompts define exactly how the Figure 8 analysis was conducted.
- chunk_07: Tables 4 and 5 appear embedded within the Appendix C.3 content of this chunk.

---

### chunk_10_bibtex_references.md
**Content:** The full `colm2025_conference.bib` file containing all 70+ BibTeX entries cited in the paper.

**Key citation groups:**
- **SWE-bench family:** jimenez2024swebenchlanguagemodelsresolve, yang2024swebenchmultimodalaisystems, openai2024swebenchverified, swebenchpro2025, zan2025multiswebench, zhang2025swebenchgoeslive, rashid2025swepolybenchmultilanguagebenchmarkrepository.
- **SWE-agents and scaffolds:** yang2024sweagentagentcomputerinterfacesenable, wang2025openhandsopenplatformai, xia2024agentlessdemystifyingllmbasedsoftware, yang2023intercodestandardizingbenchmarkinginteractive.
- **SWE-agent training data:** pan2025trainingsoftwareengineeringagents, jain2025r2egymproceduralenvironmentshybrid, yang2025swesmithscalingdatasoftware, pham2025swe.
- **Code benchmarks:** chen2021evaluating, austin2021programsynthesislargelanguage, hendrycks2021measuringcodingchallengecompetence, liu2023codegeneratedchatgptreally, jain2024livecodebenchholisticcontaminationfree, zhuo2025bigcodebenchbenchmarkingcodegeneration, mundler2024swtbench.
- **Performance optimization benchmarks:** du2024mercurycodeefficiencybenchmark, liu2024evaluatinglanguagemodelsefficient, waghjale2024ecco, huang2025effibenchbenchmarkingefficiencyautomatically, he2025sweperflanguagemodelsoptimize, ouyang2025kernelbenchllmswriteefficient, press2025algotunelanguagemodelsspeed, shetty2025gsochallengingsoftwareoptimization.
- **Game-playing AI:** silver2016mastering, openai2019dota2largescale, mnih2015human, yao2020calmexplorelanguagemodels, hu2025gamearenaevaluatingllmreasoning, karten2025pokechampexpertlevelminimaxlanguage, paglieri2025balrogbenchmarkingagenticllm, zhang2025videogamebenchvisionlanguagemodelscomplete.
- **Self-improving agents:** wang2025huxleygodelmachinehumanlevelcoding, zhang2025darwingodelmachineopenended.
- **Model cards (tested LMs):** modelcardclaude45sonnet, modelcardclaude4sonnet, modelcardopenaigpt5, modelcardopenaio3, comanici2025gemini, qwen3technicalreport, grokcodefast12025.
- **Arena citations:** battlecode2025, chung2020battlesnake, corewar1984, halite2016, huskyholdem2025, hartness2004robocode, robotrumble2020.
- **Evaluation / statistical methods:** elo1967proposed, bradley1952rank, bai2022traininghelpfulharmlessassistant, boubdir2024elo, chiang2024chatbotarenaopenplatform, herbrich2006trueskill, fudenberg1991game.
- **Agent reasoning:** yao2023reactsynergizingreasoningacting.
- **Code similarity:** ratcliff1988pattern.
- **RL/self-play:** zelikman2022star.
- **Domain applications (future arenas):** yang2023languagehackers, zhang2024cybench, abramovich2025enigmainteractivetoolssubstantially, shi2024ehragent, hou2025enhancing, bibri2020emerging.
- **TroVE:** wang2024troveinducingverifiableefficient.

**Correlates with:**
- chunk_01, chunk_02, chunk_03, chunk_04, chunk_05, chunk_06: All in-text citations appear throughout every main section; this chunk provides the full bibliographic data for every citation key.
- chunk_08: Arena citations (battlecode2025, chung2020battlesnake, corewar1984, halite2016, huskyholdem2025, hartness2004robocode, robotrumble2020) correspond directly to the 7 arena cards.

---

## Cross-File Correlation Summary

### For Reading the Paper Sequentially
The natural reading order is: chunk_01 → chunk_02 → chunk_03 → chunk_04 → chunk_05 → chunk_06 → chunk_07 → chunk_08 → chunk_09 → chunk_10.

### For Understanding the Benchmark Design
Combine: chunk_02 (formulation) + chunk_08 Appendix A (infrastructure) + chunk_09 Appendix C (agent config, tournament config).

### For Understanding Experimental Results
Combine: chunk_03 (experimental setup) + chunk_04 (results) + chunk_07 (all tables) + chunk_09 Appendix D.1–D.2 (extended results and ablations).

### For Understanding Model Failure Modes
Combine: chunk_05 (analysis section) + chunk_09 Appendix D.3–D.4 (LM judge methodology, additional analyses) + chunk_09 Appendix D.5 (verbatim judge prompts).

### For Understanding Each Arena in Depth
Use: chunk_08 Appendix B (arena cards) + chunk_07 Table 2 (arena summary) + chunk_03 (languages and constraints) + chunk_09 Appendix C.1 (arena description prompt).

### For Reproducing Elo/Statistical Analysis
Use: chunk_03 (high-level Elo description) + chunk_09 Appendix C.3 (full Bradley-Terry MLE formulation, covariance matrix, bootstrapping procedures) + chunk_07 Tables 4 and 5 (outputs).

### For Adapting to a New Domain (e.g., Physical AI Oncology Trials)
The following structural mappings are relevant:
- **Code arena** → Clinical trial simulation environment
- **Player (LM + ACI)** → AI agent managing a treatment protocol or data pipeline
- **Edit phase** → AI-driven protocol optimization or code revision round
- **Competition phase** → Evaluation run against real or simulated patient outcome data
- **Log-based feedback** → Clinical outcome metrics, adverse event logs, response rate data
- **Codebase-as-memory** → Protocol documentation, historical patient data, analysis scripts retained across rounds
- **Self-directed improvement** → Autonomous hypothesis generation and protocol refinement
- **Win condition** → Objective clinical outcome (e.g., tumor response rate, progression-free survival)
- **Adversarial adaptation** → Comparison against competing AI-designed protocols or historical best-in-class treatments
- **Multi-arena evaluation** → Multiple cancer types, treatment modalities, or patient cohorts
- **Elo/Bradley-Terry ranking** → Relative ranking of AI-designed protocol variants

---

## Notes for Claude Code Opus 4.7

1. **Word-for-word fidelity:** All chunks preserve the original paper text exactly as written, including technical notation (LaTeX symbols are written out in plain text where possible, e.g., "C(8,2)" for binomial coefficient, "Σ" for sum).

2. **Figure references:** Figures referenced in the text (e.g., Figure 1, Figure 8, etc.) correspond to images that are not included in these text chunks. The figures are described via their captions, which are preserved verbatim.

3. **Cross-references between sections:** In-text references like "Section 5.2", "Appendix C.3", "Table 1" all have corresponding content in the appropriate chunks as mapped above.

4. **Prompt content verbatim:** The mini-SWE-agent system prompts in chunk_09 Appendix C.1 are reproduced verbatim, including formatting templates. These are directly usable as prompt templates for analogous systems.

5. **LM judge prompts verbatim:** The three evaluation judge prompts in chunk_09 Appendix D.5 are complete and directly usable for analogous trajectory evaluation tasks (e.g., evaluating AI clinical trial agent trajectories for grounded decision-making, hallucination of outcome causality, and validation of protocol changes).

6. **File not included:** The `.sty` files (colm2025_conference.sty, fancyhdr.sty, natbib.sty, colm2025_conference.bst) and all image/figure files are excluded per task instructions. The `main.bbl` compiled bibliography is also excluded; the raw `.bib` file in chunk_10 is the authoritative reference source.
