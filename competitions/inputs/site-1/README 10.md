# README — Orbit Wars Kaggle Competition: Chunked Source Files

**Prepared for:** Claude Code Opus 4.7 (1M context window)
**Purpose:** Processing three chunked source files from the Orbit Wars Kaggle competition page for use in developing a new physical AI oncology trial paper.
**Original source:** https://www.kaggle.com/competitions/orbit-wars (Apple Safari Webarchive, captured May 8, 2026)

---

## Overview of Chunking Strategy

The original Orbit Wars Kaggle competition page was decomposed into three thematically distinct markdown files. Each file preserves the original source text word-for-word without abbreviation or alteration. No images are included. The three chunks separate (1) prose narrative content, (2) all structured tables and data schemas, and (3) executable code examples, the AGENTS.md guide, and citation. This separation allows targeted retrieval: reasoning about game mechanics uses Chunk 1, data schema lookups use Chunk 2, and implementation or citation work uses Chunk 3.

---

## File Descriptions

### chunk_1_site_text.md — Prose Narrative and Competition Rules

**Contents:** All prose-form sections of the Orbit Wars Kaggle competition page, including the competition header, full Description, Evaluation methodology (including Ranking System and Final Evaluation), Timeline, Prizes, Getting Started notice, and the complete "How to Play Orbit Wars" section covering: Overview, Board Layout, Planets, Planet Types, Home Planets, Fleets, Fleet Speed, Fleet Movement, Fleet Launch, Comets, Turn Order, Combat, and Scoring and Termination. Closes with competition metadata (host, prizes, participant counts, tags).

**Key concepts contained:**
- Multi-agent game design: 1v1 and 4-player free-for-all modes
- Skill rating system: Gaussian N(μ, σ²) model, TrueSkill-style Bayesian updates
- Competition submission pipeline: 5 submissions/day limit, validation episodes, μ₀=600 initialization
- Game physics: continuous 100×100 space, sun at (50,50) with radius 10, 500-turn limit
- Planet mechanics: production (1–5 ships/turn), orbiting vs. static classification, 4-fold symmetry
- Fleet mechanics: logarithmic speed scaling, continuous collision detection, straight-line travel
- Comet mechanics: elliptical paths, spawn schedule (steps 50/150/250/350/450), temporary capture
- Turn order: 7-step sequential resolution per turn
- Combat resolution: multi-attacker priority fight, garrison subtraction, ownership flip rules
- Win condition: total ships (planets + fleets) at game end

**Correlations to other chunks:**
- All field names mentioned in prose (e.g., `planets`, `fleets`, `angular_velocity`, `comet_planet_ids`) are formally defined with types in **chunk_2_tables.md** (Observation Reference table).
- All configuration parameters named in prose (e.g., `cometSpeed`, `shipSpeed`, `sunRadius`) have their default values tabulated in **chunk_2_tables.md** (Configuration table).
- The Nearest Planet Sniper agent example in **chunk_3_references.md** directly implements the Fleet Launch and targeting logic described in this chunk.
- The fleet speed formula cited in prose (`speed = 1.0 + (maxSpeed - 1.0) * (log(ships) / log(1000)) ^ 1.5`) is cross-referenced by the Fleet Speed Lookup table in **chunk_2_tables.md**.
- The AGENTS.md Game Overview summary in **chunk_3_references.md** is a condensed restatement of the full rules in this chunk; this chunk is authoritative for rule detail.

---

### chunk_2_tables.md — Tables and Structured Reference Data

**Contents:** All formally structured data from the Orbit Wars page, including the Observation Reference table, Action Format specification, Configuration table, and eight additional derived reference tables reconstructed from structured prose: Planet Data Structure, Fleet Data Structure, Comet Group Data Structure, Planet Types Summary, Comet Spawn Schedule, Fleet Speed Lookup, 4-Fold Symmetry Coordinate Mapping, Map Generation Guarantees, and Player Starting Positions.

**Key structured data contained:**
- **Observation Reference:** 8 fields with types and descriptions — `planets`, `fleets`, `player`, `angular_velocity`, `initial_planets`, `comets`, `comet_planet_ids`, `remainingOverageTime`
- **Action Format:** move tuple `[from_planet_id, direction_angle, num_ships]` with field semantics
- **Configuration:** 6 parameters — `episodeSteps` (500), `actTimeout` (1s), `shipSpeed` (6.0), `sunRadius` (10.0), `boardSize` (100.0), `cometSpeed` (4.0)
- **Planet schema:** 7-element list with field-level semantics
- **Fleet schema:** 7-element list with field-level semantics
- **Comet group schema:** dict with `planet_ids`, `paths`, `path_index`
- **Speed reference:** 1 ship=1.0, ~500 ships=~5.0, ~1000 ships=6.0 (max)
- **Symmetry mapping:** 4-quadrant coordinate transforms from base (x,y)
- **Map guarantees:** 20–40 planets, 5–10 symmetric groups, ≥3 static, ≥1 orbiting
- **Game mode starting positions:** 1v1 diagonal (Q1/Q4), 4-player one per group

**Correlations to other chunks:**
- Every field named in **chunk_1_site_text.md** prose has its canonical type definition here; this chunk is the schema authority.
- The `agent` function in **chunk_3_references.md** reads directly from the observation fields defined in this chunk's Observation Reference table; field names must match exactly.
- The Action Format table here specifies what the `agent` function in **chunk_3_references.md** must return.
- Configuration defaults here explain the constants referenced in the fleet speed formula and board geometry discussed in **chunk_1_site_text.md**.
- The Comet Spawn Schedule table expands the spawn-step list (50, 150, 250, 350, 450) stated in prose in **chunk_1_site_text.md**.

---

### chunk_3_references.md — AGENTS.md, Code Examples, and Citation

**Contents:** The complete AGENTS.md getting-started guide (Game Overview, Your Agent, Example — Nearest Planet Sniper, Agent Convenience named-tuple usage), all code blocks for local testing (`kaggle-environments>=1.28.0`, `make()`, `env.run()`, `env.render()`), the full Kaggle CLI workflow (install, auth, find competition, accept rules, download data, submit single-file/multi-file/notebook agents, monitor submissions, list episodes, download replays and logs, check leaderboard, Typical Workflow), and the formal academic citation.

**Key implementation content contained:**
- `kaggle_environments` Python package: `make("orbit_wars")`, `env.run()`, `env.steps`, `env.render()`
- Named tuple imports: `Planet`, `Fleet`, `CENTER`, `ROTATION_RADIUS_LIMIT` from `kaggle_environments.envs.orbit_wars.orbit_wars`
- Nearest Planet Sniper: complete working Python agent (~20 lines) — reads `obs.planets`, filters by owner, computes Euclidean distance, sends `ships + 1` at `atan2` angle
- Agent convenience wrapper: full skeleton showing named-tuple planet iteration, returning `[]`
- Kaggle CLI: `pip install kaggle`, token auth (`~/.kaggle/access_token`), OAuth, env var
- Submission commands: single `.py`, multi-file `.tar.gz`, notebook kernel
- Episode/replay/log download commands with `<SUBMISSION_ID>` and `<EPISODE_ID>` placeholders
- **Citation:** Bovard Doerschuk-Tiberi, Walter Reade, and Addison Howard. Orbit Wars. https://kaggle.com/competitions/orbit-wars, 2026. Kaggle.

**Correlations to other chunks:**
- The AGENTS.md Game Overview is a condensed summary; **chunk_1_site_text.md** contains the authoritative full rules for all mechanics referenced here.
- All observation field names accessed in code (`obs.get("planets", [])`, `obs.get("fleets", [])`, `obs.get("player", 0)`, `obs.player`, `obs.planets`) correspond directly to field definitions in **chunk_2_tables.md**'s Observation Reference table.
- The action return format `[from_planet_id, angle, ships_needed]` produced by the Nearest Planet Sniper maps to the Action Format specification in **chunk_2_tables.md**.
- `kaggle-environments>=1.28.0` is the required package version for running the environment defined in **chunk_1_site_text.md** and schematized in **chunk_2_tables.md**.
- The citation here is the formal bibliographic record for the competition described across all three chunks.

---

## Cross-File Correlation Map

```
chunk_1_site_text.md  ←→  chunk_2_tables.md
  - Prose game mechanics      - Formal schemas for all
    name all observation        observation fields and
    fields, config params,      configuration params
    and data structures         named in prose

chunk_1_site_text.md  ←→  chunk_3_references.md
  - Full rules authority      - AGENTS.md is a condensed
    for all mechanics           restatement; code implements
    referenced in AGENTS.md     the mechanics described here

chunk_2_tables.md     ←→  chunk_3_references.md
  - Observation Reference     - Agent code reads exactly
    table defines the fields    these fields; Action Format
    the agent code reads        table defines what the code
                                must return
```

---

## Usage Guidance for Claude Code Opus 4.7

When processing these files for a new physical AI oncology trial paper, note the following structural properties:

**Chunk 1** is the primary narrative and conceptual layer. It describes agent behavior, decision loops, environmental dynamics, turn structure, and reward-relevant outcomes (ship counts). This is the most useful chunk for drawing analogies to agent-environment interaction patterns, sequential decision-making under uncertainty, multi-agent dynamics, and time-horizon reasoning — all of which may map to oncology trial design concepts (treatment agents, biological competition, resource allocation, survival scoring).

**Chunk 2** is the schema and parameter layer. It provides precise, typed definitions for every observable variable, action variable, and configuration parameter. This is the most useful chunk when specifying data pipelines, observation spaces, action spaces, or any formal computational model that must be implemented from scratch.

**Chunk 3** is the implementation and citation layer. It contains directly runnable code, CLI workflows, and the citable bibliographic record. Use this chunk when grounding any computational reproduction of the environment, referencing prior work, or implementing the agent loop that will serve as a structural template.

All three chunks must be read together for complete fidelity: Chunk 1 explains the why and what, Chunk 2 specifies the how (data), and Chunk 3 specifies the how (code). No single chunk is self-contained for implementation purposes.

---

## Source Metadata

| Property | Value |
|---|---|
| Source URL | https://www.kaggle.com/competitions/orbit-wars |
| Archive format | Apple Safari Webarchive (.webarchive) |
| Capture date | May 8, 2026 |
| Competition start | April 16, 2026 |
| Final submission deadline | June 23, 2026 |
| Prize pool | $50,000 (10 × $5,000) |
| Authors (citation) | Bovard Doerschuk-Tiberi, Walter Reade, Addison Howard |
| kaggle-environments version required | ≥1.28.0 |
