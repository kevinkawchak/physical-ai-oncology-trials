# Chunk 04 — Results

---

## 4. Results

We present our main results in Table 1 (Main Results Table).
`Claude Sonnet 4.5` stands at the top, followed closely by `o3` and `GPT-5`.
After a gap of 100 Elo, the next best models are `Claude Sonnet 4` and `GPT-5 mini`.
Notably, no single model across dominates all arenas.
Top ranked `Claude Sonnet 4.5` places just 4th in Poker, emphasizing the importance of CodeClash's support for multiple arenas.
Figure 2 shows win rates of specific matchups.
Figure 3 reveals distinct performance trends across rounds -- some models excel early before plateauing, while others improve steadily over time.

**Figure 2 Caption:** Model win rates (row beats column). Win rate is the proportion of tournaments (out of 240) won across all arenas. `Claude Sonnet 4.5` has the highest average win rate at 69.9%.

**Figure 3 Caption:** Win rates across rounds, illustrating how different models gain (`Claude Sonnet 4.5`) or lose momentum (`GPT-5`) over the course of the tournament.

### 4.1 Ablations

**On RobotRumble, models trail substantially behind expert human programmers.**
From RobotRumble's leaderboard (https://robotrumble.org/boards/2), we identified the top open-source submission as of October 31, 2025, a bot called `gigachad` authored by `entropicdrifter` (https://robotrumble.org/entropicdrifter/gigachad).
We run 10 tournaments of 15 rounds of `Claude Sonnet 4.5` (#1 on RobotRumble) versus `gigachad`.
Throughout a tournament, `gigachad` remains static; no human or LM optimizes it between rounds.

`Claude Sonnet 4.5` is dominated by `gigachad`, **winning exactly *zero* of the 150 rounds**.
Each round of RobotRumble, we run 250 simulations and determine the winner by majority.
Out of 150×250=37,500 simulations, `Claude Sonnet 4.5`'s code wins *zero*.
For explanations about where models fall short, we discuss in depth in Section 5 (Analysis).

Leaderboards for other arenas do not exist (Core War, RoboCode) or do not have readily open source, ranked submissions (Halite, BattleSnake, Poker).
While striking, our results admittedly are drawn from a limited sample.
We hope CodeClash can facilitate further exploration in human-AI dynamics (e.g., humans competing against evolving AI opponents, human-AI collaborative development).
Such studies require careful experimental design and recruitment that is best left as future work.

**Models have limited capacity for opponent analysis even with transparent codebases.**
For each pairwise matchup among `Claude 4.5 Sonnet`, `GPT-5`, and `Gemini 2.5 Pro`, we run 10 Core War tournaments of 15 rounds each, with one modification -- before the *edit* phase of round `n`, each player receives a read-only copy of their opponent's code from round `n-1`.
While the relative standings remain consistent with the default setting, the win rates change with `GPT-5` securing 74.6% (+7.8%) of rounds, `Claude 4.5 Sonnet` at 53.2% (-1.8%), and `Gemini 2.5 Pro` at 22.7% (-5.5%).
Curiously, `GPT-5` only accesses its opponent's codebase in 12.8% of all rounds, far fewer than `Claude 4.5 Sonnet` (99.3%) and `Gemini 2.5 Pro` (52.9%), suggesting that frequent inspection of opponent code does not necessarily translate to competitive advantage, as our analysis later in Section 5.2 reaffirms.
Additional insights in Appendix D.2 (Additional Ablations).
Subsequent studies could more thoroughly investigate and enhance models' capacity for detecting opponents' weaknesses and designing tailored counter-strategies.

**Multi-agent competitions (3+ players) reflect similar rankings.**
We run 20 Core War tournaments, 15 rounds each, with 6 of 8 models (excluding `GPT-5-mini`, `Claude 4 Sonnet`).
To quantify performance, as shown in Table 2 (TrueSkill Table), we use the TrueSkill rating system since Elo and win rate are limited to one-on-one settings.
The results are similar to Core War ranks in Table 1, with `GPT-5` and `Grok Code Fast` (two models of similar Elo ranking) switching positions.
However, the 6 player tournaments exhibit far more competitive volatility.
Lead changes (round `n` winner different from round `n-1`) occur 48.4% of the time in 6 player Core War, compared to just 18.2% in the two player setting.
Winners of 6-player tournaments capture just 28.6% of total points on average versus 78.0% in 2-player settings.
We provide some additional insights in Appendix D.
We look forward to future work that can leverage CodeClash's multi-player tournaments as a testbed for understanding strategic behaviors such as coalition dynamics, positional play, and risk management.
