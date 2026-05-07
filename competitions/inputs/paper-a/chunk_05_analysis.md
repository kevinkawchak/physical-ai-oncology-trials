# Chunk 05 — Analysis

---

## 5. Analysis

### 5.1 Competitive Dynamics

Beyond overall win rates, we analyze how models interact with their codebases along with the resilience of models after losing individual rounds.
We also investigate trends in models' solution diversity and codebase organization across tournaments.

**Figure 4 Caption:** Probability of winning the next round after losing several rounds in a row. Even the highest ranking models struggle to recover after losing one or more consecutive rounds in a tournament. Numbers in parentheses indicate the overall average win rate.

**Figure 5 Caption:** To measure solution diversity, we compute code similarity of each model's solutions to itself at the same round. Each data point represents the mean pairwise similarity between a model's solution (`main.py`) at round `n` across 70 BattleSnake tournaments.

**Models interact with codebases in markedly different ways.**
CodeClash's open-ended setting reveals striking differences in how models operate in the *edit* phase.
For instance, while `o3` and `Gemini 2.5 Pro` typically only edit an average of 2 files per round, `GPT-5` usually changes 5 to 6.
The size of edits also varies -- on one end, `o3` typically adds/removes a total of 51 lines per round, 8× less than `Qwen3 Coder` or the Claude Sonnet family which usually modify more than 400 lines.
`Gemini 2.5 Pro` stands out as a verbose thinker, generating an average of 105 words per thought, more than double the average.
`Claude Sonnet 4.5` usually takes 23 of the allotted 30 editing turns per round, whereas `GPT-5` and `o3` typically concludes after just 15 steps.
Distributions visualizing these tendencies in Appendix D.1 (Interaction Trends).

Intriguingly, we did not find any correlations between any of these behaviors and win rates.
Both minimalists (`o3`) and high activity editors (`Claude 4.5 Sonnet`) succeed.
Compared to existing benchmarks that terminate upon reaching a solution, CodeClash's multi-round competitive setting makes these distinctions even more salient.

**Even strong models struggle to recover after losing rounds.**
In real-world software development, early choices are often made under uncertainty: the best approach might only become clear after testing, real world deployments, and observing competitors.
Therefore, the ability to interpret noisy signals and to reconsider internal hypotheses and core design decisions is an important factor in real-world success.
The round-based nature of CodeClash exposes how poorly LMs adapt once their initial strategies fail.
Figure 4 shows that even for the `Claude Sonnet 4.5`, losing a single round results in a comeback probability (win probability of the next round) of less than one third — less than half of the overall round win rate of 71%.
For `o3`, the win rate drops to only 26% after a single loss (compared to an overall round win rate of 65%).
After five consecutive defeats, comeback rates fall below 15% for `Claude Sonnet 4.5`, and below 10% for all other models.
This suggest an inability of models to reconsider strategies, or adapt to opponents or the arena state.

**Models' solutions become increasingly diverse with every round.**
For each [model, opponent, round] tuple, we compute code similarity across the model's solutions (10 samples) using Python's `difflib.SequenceMatcher`.
In other words, we have 10 tournaments of `Claude Sonnet 4.5` vs. `o3` from our main results.
We then compute a similarity matrix between all 10 versions of `Claude Sonnet 4.5`'s `main.py` at each round 1/5/10/15, and finally calculate a mean similarity score.
We run this analysis just for the BattleSnake arena since solutions are written in Python in a single `main.py` file.
From Figure 5, we observe models' solutions generally become more dissimilar with every round.
Each round, models are attempting to not only make absolute improvements, but also adapt to opponent play.
Solution diversity varies with model (`o3` at 0.63 versus `GPT-5` at 0.41 at round 1), though the effect of the opponent's identity is less pronounced, as we show in Appendix D.3 (Additional Analyses).
Unlike existing code benchmarks where models quickly converge on canonical solutions, CodeClash elicits substantial creativity from models, even against the same opponent.
This diversity makes CodeClash a potentially effective training ground for improving models via self-play and reinforcement learning.

**Figure 6 Caption:** The total number of created files scales almost linear with the round. R refers to the filename redundancy at round 15; high values indicate repeating patterns in filenames (such as `main1.py`, `main2.py`, ...).

**Figure 7 Caption:** Models differ in the average number of *throwaway files* (files not used after the round in which they were created). The stacked bars distinguish between files at the repository root and those in subdirectories.

**Codebases managed by models become messier over time.**
In most human-managed codebases, the rate of file creation quickly plateaus once the overall structure has been established; subsequent work primarily focuses on refinement, maintenance, and incremental improvements rather than continuous expansion.
In contrast, we observe a markedly different trend in Figure 6: the average number of agent-created files scales almost linearly with the number of rounds.
`Claude 4.5 Sonnet` exhibits the highest file creation activity, averaging more than 30 files per tournament, followed by `GPT-5` (21), whereas `o3` creates fewer than 5.
For `Claude Sonnet 4.5`, the high average is driven by consistent creation of various files at the repository root (making the codebase even less orderly); for `GPT-5`, the average is elevated by tournaments that accumulate particularly many output and temporary files in separate directories that were never cleaned up.
These observations again highlight how the top three models interact with their codebases in distinctly different ways.

When many files are produced, filenames often become repetitive and follow systematic patterns (e.g., `analyze_round_13_v2.py`).
We quantify this effect through the *filename redundancy* metric (the fraction of files sharing name prefixes with other files) which is particularly high for `Qwen 3 Coder` (59%) and the `Claude Sonnet` models (35%).
In addition, most agent-created files are never referenced, reused, or modified in subsequent rounds.
We quantify these *throwaway files* in Figure 7: `Claude 4.5 Sonnet` (18 files per tournament) and `GPT-5` (15) again rank at the top, whereas `o3` remains near the bottom.

Together, Figure 6 and Figure 7 reinforce the view that most LMs struggle to converge toward maintainable file structures over time, favoring the continual generation of new, often redundant scripts over the systematic refinement and reuse of existing code.
We include more graphs along with case studies of specific codebases in Appendix D.3 (Additional Analyses).

### 5.2 Strategic Reasoning Limitations

We investigate models' capacity for self-improvement by analyzing how they interpret competition results to diagnose failures, decide what code changes to make, and how to validate them.
This analysis is performed using GPT-5 with high reasoning as a judge.
Details, as well as additional analyses of agent trajectories in terms of the nature of actions, are presented in Appendix D.3 (Analyzing trajectories using LMs as a judge).

**Figure 8 Caption:** LMs struggle to analyze log files from previous rounds and frequently hallucinate about why rounds were lost. Using LM-as-a-judge, we annotate players' trajectories with answers to three questions (a) Are changes to solutions grounded in the analysis of previous rounds or testing? (b) Are there hallucinated or unsubstantiated claims about why a round was lost? (c) Are changes validated by arena simulations or unit tests?

**Most models struggle to interpret logs or derive meaningful insights about their performance.**
Agents have access to detailed log records of all previous rounds, encompassing several hundred to thousands of runs against their opponent.
These logs can not only reveal whether the last round's changes improved the winning rate, but detail the exact behavior that led to losses or wins.
However, despite explicit suggestions to write analysis tooling in the prompt, most LMs do not manage to extract meaningful information, often stopping at reading the first lines of a log file, or calculating the win rate of the last round.
Figure 8 (a) shows whether the combined output of the actions of the agent (i.e., the entirety of the information available to the agent) could motivate the edits performed by the agent.
While most edits of the `Claude Sonnet` models can be motivated in this way, the edits of all other models are ungrounded in more than 65% of all rounds.
Interestingly, `o3` scores particularly low in this aspect, with ungrounded edits in almost 80% of rounds.

**Models hallucinate during failure analysis and misinterpret logs and analysis outputs.**
The most salient pattern are agents infering causal explanations for arena outcomes after reviewing only the opening lines of a single log file, when these lines do not even show the deciding moment in an arena.
Behaviors of this kind are quantified in Figure 8 (b).
For example, `Claude Sonnet 4.5` makes uncorroborated claims about the exact reason a game was lost in more than 17% of rounds on average.
However, this behavior is much more pronounced in certain arenas, such as BattleSnake, where `Claude Sonnet 4` and `Claude Sonnet 4.5` hallucinate about loss causality in 34% and 46% of rounds.
Most hallucinations are misinterpretations or over-interpretations of log files and similar outputs, though claims that cannot be connected to any source also occur.

**Models make changes without assessing their effects.**
When models propose algorithmic changes, they seldom confirm whether modifications work as intended or if the new solution outperforms previous iterations.
The prompt explicitly suggests running arena simulations between different versions of code or writing unit tests to validate intended behavior.
Combining exploratory methods with self-play could likely avoid unwanted regressions.
Nevertheless, most models deploy untested code. As shown in Figure 8 (c), only Claude Sonnet 4.5 validates changes in a majority of rounds (56%), followed by GPT-5 (50%), whereas Gemini 2.5 Pro and o3 perform meaningful validation in one out of five rounds.

**Models rarely make bash mistakes.**
Across all models, more than 85% of generated actions execute successfully, with error rates ranging from just 10% (`Claude Sonnet 4`) to 16% (`Qwen3 Coder`).
Models also recover rapidly from errors: following a failed command, the very next action runs successfully more than 80% of the time.
This stands in stark contrast to earlier findings of "cascading failures" in agent systems, suggesting command-line proficiency has improved substantially in recent models.
These results indicate that performance differences in CodeClash stem from strategic reasoning and code quality, not `bash` interface capabilities.
More graphs confirm this strength in Appendix D.1 (Interaction Trends).
