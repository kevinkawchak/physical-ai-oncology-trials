# Chunk 03 — Experiments

---

## 3. Experiments

**Models.** We select 8 strong LMs to evaluate, where strength is roughly estimated as performance on existing coding benchmarks.
Our final list includes two models from the Anthropic family (Claude Sonnet 4.5, Claude Sonnet 4), three models from the OpenAI family (GPT 5, GPT 5-mini, o3), Gemini 2.5 Pro, Qwen3-Coder, and Grok Code Fast 1.

**Agent system.** As discussed in Section 2.2, we use `mini-SWE-agent`.
We intentionally decide against using tool-heavy scaffolds such as SWE-agent or OpenHands, as they are often optimized for models and benchmarks.
By restricting interactions to bash commands, `mini-SWE-agent` avoids imposing predefined assumptions via tools about how LMs should approach codebase modifications or competitive play.
Per round, models are allotted a maximum of 30 turns for the *edit* phase, with automatic termination if exceeded.
Player configurations are discussed thoroughly in Appendix C.1 (mini-SWE-agent Configuration).

**Number of rounds run.** For our main leaderboard, we make models compete one-on-one.
Given 8 models and 6 arenas, we run 10 tournaments per model pair per arena, with each tournament lasting 15 rounds.
This yields C(8,2) × 6 × 10 × 15 = 25,200 total rounds.
Tournament runtime varies by arena, taking 75 minutes on average -- totaling 2.4 million hours of runtime (mostly due to model latency), parallelized over the independent tournaments.
Tournament configuration details are covered in Appendix C.2 (Tournament Configuration).

**Win rates.** Performance per model is generally calculated as an aggregation across all tournaments (sets of 15 rounds) won across all arenas.
A single round is won by a model if it achieves a higher score in the arena than its opponent or if its opponent makes an invalid submission.
A tournament is won by the model that wins more rounds than its opponent, or, if both models win equally many rounds, by the model that scores the last win.
(Draws are a possible outcome for each round, so both models might achieve an equal number of wins in a tournament. In the very rare event of a tournament consisting only of draw rounds, the tournament is considered a draw.)
The win rate of a model is the fraction of tournaments it has won.
For details, see Appendix C.3 (Evaluation Metrics).

**Elo metrics.** Inspired by the thread of prior work ranking LMs on the task of instruction following, we use Elo scores with a base rating of R=1200 and a slope of 400 to quantify the overall strength of each model.
Instead of calculating Elo scores using sequential updates (which require a choice of step size and depend on update order), we perform a more rigorous maximum likelihood fit to the win rates.
We validate rank stability and our statistical treatment with both parametric and non-parametric bootstrapping experiments and observe more than 98% pairwise order agreement.
For details, see Appendix C.3 (Evaluation Metrics).
