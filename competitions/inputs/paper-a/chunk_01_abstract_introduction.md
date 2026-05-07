# Chunk 01 — Abstract and Introduction
**Paper:** CodeClash: Benchmarking Goal-Oriented Software Engineering
**Authors:** John Yang*, Kilian Lieret*, Joyce Yang, Carlos E. Jimenez, Ofir Press, Ludwig Schmidt, Diyi Yang
**Affiliations:** Stanford University, Princeton University, Cornell University

---

## Abstract

Current benchmarks for coding evaluate language models (LMs) on concrete, well-specified tasks such as fixing specific bugs or writing targeted tests.
However, human programmers do not spend all day incessantly addressing isolated tasks.
Instead, real-world software development is grounded in the pursuit of high-level goals, like improving user retention or reducing costs.
Evaluating whether LMs can also iteratively develop code to better accomplish open-ended objectives without any explicit guidance remains an open challenge.
To address this, we introduce CodeClash, a benchmark where LMs compete in multi-round tournaments to build the best codebase for achieving a competitive objective.
Each round proceeds in two phases: agents edit their code, then their codebases compete head-to-head in a code arena that determines winners based on objectives like score maximization, resource acquisition, or survival.
Whether it's writing notes, scrutinizing documentation, analyzing competition logs, or creating test suites, models must decide for themselves how to improve their codebases both absolutely and against their opponents.
We run 1680 tournaments (25,200 rounds total) to evaluate 8 LMs across 6 arenas.
Our results reveal that while models exhibit diverse development styles, they share fundamental limitations in strategic reasoning.
Models also struggle with long-term codebase maintenance, as repositories become progressively messy and redundant.
These limitations are stark: top models lose every round against expert human programmers.
We open-source CodeClash to advance the study of autonomous, goal-oriented code development.

*Equal contribution. Correspondence to johnby@stanford.edu, kl5675@princeton.edu.*

---

## 1. Introduction

Existing coding benchmarks challenge language models (LMs) to complete small, focused tasks, such as implementing an algorithm, fixing a specific bug in a single function, or writing a test for a target class.
Problem statements are straightforward and fine-grained in their description of a task.
Given explicit instructions, models are evaluated on their ability to execute them correctly.

On the contrary, real world software development demands a much broader scope of agency.
Instead of maintenance tasks, developers are driven by high-level goals like improving user retention, increasing revenue, or reducing costs.
This requires fundamentally different capabilities; engineers must recursively decompose these objectives into actionable steps, prioritize them, and make strategic decisions about which solutions to pursue.
The process is a continuous loop -- propose changes, deploy them, analyze real-world feedback (e.g., metrics, user behavior, A/B test results), and repeat to inform the next move.
Evaluating how models fair under such conditions remains an unaddressed challenge in benchmarking.

Therefore, we introduce CodeClash, a benchmark for goal-oriented software engineering.
Specifically, multiple LM systems compete to build the best codebase for achieving a high-level objective over the course of a multi-round tournament.
These codebases implement solutions that compete in a code arena, such as BattleSnake (grid-based survival), Poker (no-limit Texas Hold'em), and RoboCode (tank combat).
Crucially, LMs do not play directly, unlike existing game-based benchmarks.
Instead, they iteratively refine code that competes as their proxy.

As shown in Figure 1, each round proceeds in two phases: agents edit their code, then their codebases compete head-to-head in a code arena.
The code arena then executes multiple implementations against one another and determines winners based on objectives like score maximization, resource acquisition, or survival.

Success in CodeClash requires models to determine their own improvement strategies.
From the outset, LM agents receive only a brief description of the setting.
While information like arena mechanics, example bots, and recommended strategies are available in the starter codebase, models must take initiative to proactively discover them.
Each round, LMs receive gigabytes of logs from past rounds, which they can parse to extract insights about outcomes and opponents -- or ignore entirely.
Across the span of a tournament, CodeClash reveals whether and how models populate their codebases with notes, tests, and analyses.

We evaluate 8 frontier LMs across 6 arenas.
We find CodeClash elicits substantial creativity from models; across 1680 tournaments, we observe that a model's solutions become increasingly dissimilar round over round, even when facing the same opponent in the same arena.
However, our results reveal that while models exhibit diverse development styles, they share common limitations in interpreting competitive feedback, validating changes, and maintaining organized codebases over time.
Even top models hallucinate reasons for failure or modify code without confirming if these changes meaningfully improve performance.
A substantial gap remains between model and human performance; the best model (Claude Sonnet 4.5) fails to win a single round against an expert human-written bot.

We release CodeClash as an open source toolkit, including the code, arena logs, and a leaderboard, to further the study of self-evolving, LM-based SWE-agents.

**Figure 1 Caption:** CodeClash is a benchmark where players (LMs as SWE-agents) compete in programming tournaments spanning multiple rounds. Per round, models edit their codebases (edit phase) before the codebases face off in a code arena (competition phase). Then, the competition logs are copied back into the codebases and the next round begins.
