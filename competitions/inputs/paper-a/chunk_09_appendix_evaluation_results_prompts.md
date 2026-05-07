# Chunk 09 — Appendix C: Evaluation, Appendix D: Extended Results, and LM Judge Prompts

---

## Appendix C: Evaluation

In this section, we provide additional details about our evaluation procedure, including inference services, `mini-SWE-agent` configurations, arena-specific prompts, and formulae for calculating win rate and Elo scores.

### C.1 `mini-SWE-agent` Configuration

The `mini-SWE-agent` ACI allows one to define a number of configurations (https://mini-swe-agent.com/latest/advanced/global_configuration/).
We highlight a couple of configuration settings relevant to the evaluation set up for CodeClash.

**Turn and cost limits.** For the *edit* phase of each round, the LM is constrained to at most 30 interactive turns with the codebase.
We also impose a $1 cost limit, meaning once the running cost of input and output tokens for a single round exceeds $1, the editing episode is automatically terminated.
Consequently, this means that for a tournament of `n` rounds, at most $n are spent per player.
We enforce this cost limit not only to keep expenses manageable but also to discourage degenerate behaviors such as the model dumping entire files into its context, repeatedly echoing large outputs, or otherwise flooding the interaction buffer with irrelevant information.
Generally, the limit forces the agent to allocate its context budget carefully, encouraging concise reasoning and selective use of code.
We set the `mini-SWE-agent` configuration to the following values to enforce these practices:
- The `step_limit` is set to 30. The `cost_limit` is set to 1.
- In the `action_observation_template`, a prompt template that environment observations are interpolated into, the agent is reminded of the number of turns and cost consumed with the line: `<limit_note>This is the output of step {{n_model_calls}} ({{step_limit}} limit). You've used {{model_cost | round(2)}} USD ({{cost_limit}} USD limit).</limit_note>`

We observe in practice that the cost limit is almost never reached.
On the other hand, turn limits are exhausted frequently for specific models.

**Setting the context.** The system prompt briefly sets the context and informs the model of the general nature of the setting it's operating in.
Here is the prompt verbatim:

**System Prompt:**
```
You are a helpful assistant interacting continuously with a computer by submitting commands.
You'll be editing a codebase to play a programming game.

<important>
This is an interactive process where you will think and issue ONE command, see its result, then think and issue your next command.
</important>

Your response must contain exactly ONE bash code block with ONE command (or commands connected with && or ||).
Include a THOUGHT section before your command where you explain your reasoning process.
Format your response as shown in <format_example>.

<format_example>
Your reasoning and analysis here. Explain why you want to perform the action.
```bash
your_command_here
```
</format_example>

Failure to follow these rules will cause your response to be rejected.
```

The LM is informed it is acting in the role of a software developer with the ability to investigate and edit a codebase across multiple turns.
The prompt clearly delineates an interaction protocol.
Every turn, the model should be explaining its reasoning in a "Thought" section, followed by a `bash` code block.

**Describing the arena and tournament.**
After the system prompt, the next message given to the LM briefly describes the arena and thoroughly reviews how the LM can interact with the codebase environment correctly.
We first show the arena description:

**Subsection of initial message describing the arena:**
```
## Game Description

{{game_description}}

## General tips about how to play the game

The details of the game are fully available within this codebase.
- `docs/`: Game documentation
- `logs/`: Past rounds and outcomes
- `trajs/`: History of your edits
- and a lot more. It's up to you to explore and utilize these resources.

The game is played in rounds and you will be evaluated on the performance over all the rounds. You won't remember past rounds.

In every round, you have a limit of {{step_limit}} steps and a cost limit of {{cost_limit}} dollars.
We will show you the number of steps and cost used so far after every response in the `<limit_note>` tag.
After you've reached the step or cost limit, you cannot continue working on this task, and we will play the game with your codebase.
This means that it's fine to reach the step or cost limit while working on documentation or testing, but you shouldn't
reach the limit while working on the actual game logic to avoid submitting an invalid codebase.

So if you want to carry knowledge forward — leave tools, notes, or strategies in the codebase.
Good documentation means you (and others) can pick up right where you left off.

If you'd hate to repeat a step next round, encode it now — as a script, a note, or a tool.

Improve the bot however you like — experiment, document, iterate. Some ideas:
- Build analysis tools
- Create bot variants to test
- Track strategies across rounds
How you choose to evolve and document is up to you. Good luck!
```

The actual description of the arena, represented by `game_description`, is brief.
These are filled in by the system templates shown in the arena cards of Appendix B (Arenas).
This lack of detail is intentional.
We impose the burden of understanding how exactly an arena works.
With full access to documentation and logs in the codebase, CodeClash forces LMs to identify and fill in gaps about its understanding of the game.
This obstacle is realistic.
As prior work around coding evaluations has demonstrated, real world software issues are often ambiguous and abstract on face value.
CodeClash enables investigating whether models can address such uncertainty by placing it in a setting where information is available, but not immediately obvious.

The second half of the prompt states the available assets, then reminds the model of both the step/cost limit along with the transient nature of its memory.
The model is explicitly informed that its working memory is *not* retained across rounds, so it is encouraged to use the codebase to maintain long-term information, tools, and general progress.
Collectively, the prompt incorporates the challenges discussed in Section 2.3 (Features).

**Subsection of initial message describing interaction:**
```
## Command Execution Rules

You are operating in an environment where

1. You write a single bash command
2. The system executes that command in a subshell
3. You see the result
4. You write your next command

For each of your response:

1. Include a THOUGHT section explaining your reasoning and what you're trying to accomplish
2. Provide exactly ONE bash command to execute
3. The action must be enclosed in triple backticks (see below for formatting rules)
3. Directory or environment variable changes are not persistent. Every action is executed in a new subshell.
   However, you can prefix any action with MY_ENV_VAR=MY_VALUE cd /path/to/working/dir && ... or write/load environment variables from files

Format your responses like this:

<format_example>
THOUGHT: Here I explain my reasoning process, analysis of the current situation, and what I'm trying to accomplish with the command below.

```bash
your_command_here
```
</format_example>

Commands must be specified in a single bash code block:
```bash
your_command_here
```

**CRITICAL REQUIREMENTS:**
- Your response SHOULD include a THOUGHT section explaining your reasoning
- Your response MUST include EXACTLY ONE bash code block
- This bash block MUST contain EXACTLY ONE command (or a set of commands connected with && or ||)
- If you include zero or multiple bash blocks, or no command at all, YOUR RESPONSE WILL FAIL
- Do NOT try to run multiple independent commands in separate blocks in one response
- Directory or environment variable changes are not persistent. Every action is executed in a new subshell.
- However, you can prefix any action with MY_ENV_VAR=MY_VALUE cd /path/to/dir && ... or write/load environ variables from files
```

We omit the examples of proper, well-formed interactions following this prompt.
The examples include actions such as how to edit a file with `sed`, performing searches of the codebase with `grep` and `find`, and viewing specific parts of files with `nl`.
We observe both with this work and prior evaluations that including such in-context demonstrations is meaningfully helpful to reducing the errant actions issued by a model.
All players' codebases are initialized with no tools provided upfront.
However, throughout the course of a tournament, models are free to synthesize their own scripts and aliases.

**Errant action handling.**
Last but not least, in the case that a model does issue an invalid action, we inherit the guardrail and error handling principles described in prior work and inform the model of such errors.
The `format_error_template` is shown when the model's response does not abide by the ReAct style form factor requested, and the following error message is displayed:

**Format error template:**
```
Please always provide EXACTLY ONE action in triple backticks, found {{actions|length}} actions.
If you want to end the task, please issue the following command: echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
without Any other command.
Else, please format your response exactly as follows:

<response_example>
Here are some thoughts about why you want to perform the action.

```bash
<action>
```
</response_example>

Note: In rare cases, if you need to reference a similar format in your command, you might have
to proceed in two steps, first writing TRIPLEBACKTICKSBASH, then replacing them with ```bash.
```

Note that the error template is *not* thrown if the action itself is problematic or executes with a non-zero return code.
This message is only invoked when the model's response doesn't abide by the expected format, and it does not account for any syntax issues or execution outcomes related to the `action` itself.

### C.2 Tournament Configuration

In addition to configuring interaction, we also allow users to set tournament settings, such as game mechanics and rounds, via a configurable `.yaml` file as well.

**Tournament configuration file for Battlesnake:**
```yaml
tournament:
  rounds: 25
game:
  name: BattleSnake
  sims_per_round: 1000
  args:
    width: 11
    height: 11
    browser: false
```

The configuration file contains two sections.
The `tournament` field allows one to specify how many `rounds` the tournament will be played.
The `game` field indicates which code arena the tournament is being played in.
`sims_per_round` is the number of simulations run per round in order to determine a winner (usually 1000).
For most games, a simulation is run by calling an executable or script with arguments.
The `args` field is a way to pass in flags to that executable to adjust the configurations of the arena.
For instance, in the above example, the `args` are eventually interpolated into the following command to run the game: `python main.py --width 11 --height 11 --browser false`.

**Player configuration section:**
```yaml
players:
- agent: mini
  name: p1
  config:
    agent: !include mini/default.yaml
    model:
      model_name: openai/gpt-5-mini
- agent: mini
  name: p1
  config:
    agent: !include mini/default.yaml
    model:
      model_name: anthropic/claude-sonnet-4-20250514
```

The player configuration is simple, essentially serving as a meta-configuration for creating each player as an LM along with a `mini-SWE-agent` configuration.
Using this configuration, it is possible to equip models with different prompts by swapping out the `mini-SWE-agent` configuration (`!include mini/default.yaml`), although we do not do this for our main leaderboard and results unless specified as otherwise.

**Number of rounds run.** To determine the number of tournaments and rounds to run to obtain a statistically meaningful leaderboard, we identify several parameters.

- *M* for the number of models to evaluate.
- *A* for the number of arenas we want models to compete in.
- *T* for the number of tournaments we run per arena.
- *P* for the number of players per tournament.
- *R* for the number of rounds per tournament.

Given these values, we can generally calculate the number of rounds that would be run with C(M,P) × A × T × R.
This assures us that each model is run against other models on the same set of arenas for the same number of total rounds (T × R).
The main results table reflects values of M=9; A=6; T=10; P=2; R=15, giving us a total of 32,400 total rounds run, with each model playing a total C(M-1,P-1) × A × T × R = 7200 rounds.
For the Section 4.1 (Ablations) evaluation with 3+ players, we use the same calculation to determine number of tournaments to run.

### C.3 Evaluation Metrics

This section contains detail on the evaluation metrics, in particular the Elo ratings for each model.
Detailed statistical analysis shows that the ranking is stable.
For example, the pairwise order agreement of our ranking is more than 98% in bootstrapping experiments.

#### C.3.1 Definitions

**Tournaments** are a sequence of 15 rounds played in one arena between two or more models.

**Winning a round.**
A round consists of one or more repetition of an arena between the submissions of different models.
A round is won by a model if any of the following applies:
1. The model is the only one with a valid submission (for example because the other model's submission does not compile or execute)
2. The model scores higher than all others. Scores are typically either win rates (across all repetitions of the arena), or other aggregate quantities (e.g., total amount of money won in poker).

**Winning a tournament:**
A tournament is won by the model that wins more rounds than its opponent, or, if both models win equally many rounds, by the model that scores the last win.
If all rounds of the tournament are draws, then the tournament is a draw (an extremely rare occurrence, less than once per 1000 tournaments).

**Win rate** per model is the fraction of tournaments won.
This metric can be further stratified into arena and opponent-specific percentages.

**Elo rating.**
We quantify absolute model strengths by Elo ratings.

Elo ratings are based on the Bradley-Terry model that models win probabilities between two players i and j with strengths s_i and s_j via logistic regression of the strength difference s_i - s_j, i.e.,

P(model i wins over j) = 1 / (1 + exp(s_i - s_i')) = σ(s_i - s_i').

Repetitions of independent games are Bernoulli-distributed and the optimal values of s_i and s_j can be calculated using a maximum likelihood fit to the win numbers w_ij (number of times i won over j), i.e.,

log L = Σ_{i<j} [w_ij * log σ(s_i - s_j) + w_ji * log σ(s_j - s_i)].

However, this leaves a gauge freedom in the strengths s_i, because all s_i can be shifted by a constant factor s_i → s_i + S without changing the value of L.
To fully constrain the fit, we choose Σ_i s_i = 0.
This choice only results in a fixed offset for the final Elo scores.
Log likelihood profiles for a fit to all arenas are found in Figure elo:ll_fit_validation_plot.

The player strengths can be converted to Elo scores R_i as:

R_i = R_0 + (β / log 10) * s_i

Following the conventions from Chess, we choose a starting Elo of R_0 = 1200 and a slope of β = 400.
Note that this convention is merely a presentation choice that affects readability, not the model predictions (unlike the K factor that is used in sequential calculation of Elo scores).

#### C.3.2 Statistical Uncertainties

The covariance matrix Σ of the player strengths s_i is given by the inverse of the Hessian matrix of log L.
Setting p_ij = σ(s_i - s_j) and n_ij = w_ij + w_ji, the Hessian of L is given by:

H_ij = ∂² log L / (∂s_i ∂s_j) = -Σ_{i<j} n_ij * p_ij * (1 - p_ij) * { 1 if i=j, -1 if i≠j }

However, this Hessian is singular, due to the above mentioned shift-invariance.
So we invert H in the constrained subspace of our gauge, S = {s_i | Σ_i s_i = 0}, i.e., calculate the covariance Σ as:

Σ = Z (Z^T H Z)^{-1} Z^T,

where Z projects onto S and is given by:

Z_ij = { 1 - 1/n if i=j, -1/n if i≠j }

The variance of s_i is then given by Var(s_i) = Σ_ii and can readily be scaled to the variance on R_i.
The uncertainties of the final results are shown in Table 5 (Elo ratings with uncertainties).

#### C.3.3 Statistical Validation and Rank Stability

We perform non-parametric and parametric bootstrapping experiments to test the stability of the ranking.
The statistical uncertainties derived from the bootstrapped Elo results agree well with those calculated from the Hessian matrix in Table 5.
Various rank stability metrics are shown in Table 4 (Rank stability metrics).
In particular, we'd like to highlight that the pairwise order agreement of our ranking is 98%.

**Non-parametric bootstrapping:**
We perform a non-parametric bootstrapping experiment by sampling with replacement from all tournaments.
This results in new win counts w_ij from which we can calculate new Elo rankings R_i.
We draw 1000 samples and calculate rank stability metrics and uncertainties based on the 1000 corresponding Elo rankings.

**Parametric bootstrapping:**
We generate bootstrap replicas from the fitted Bradley-Terry model, i.e., we use the Bradley-Terry player strengths ŝ_i that maximize the log likelihood and assume win probabilities:

p*_ij = σ(ŝ_i - ŝ_j).

For each observed matchup (i,j) with n_ij = w_ij + w_ji total games, we then draw:

w̃_ij ~ Binomial(n_ij, p*_ij), w̃_ji = n_ij - w̃_ij.

This preserves the observed matchup graph and game counts while sampling outcomes according to the fitted model.
From each resampled win matrix we refit the Bradley-Terry model (and convert to Elo) and assess variability of scores and ranks across 1000 replicas.

---

## Appendix D: Extended Results

In this section, we present additional analyses and findings not presented in Section 4 (Results).
These insights further characterize model behavior and performance in the CodeClash setting.

### D.1 Interaction Trends

We provide additional analyses and visualizations revealing trends in how different models interact with their codebase environment, such as how many steps they take per round, the size and frequency of their edits, and their length of their thoughts.

**Models differ in the number of files created or edited.** As shown in figures on CDF of files edited per round and average lines changed per round, we observe that models vary significantly in the number of files and lines changed per round.
The range varies significantly, with more conservative models such as `o3` or `Gemini 2.5 Pro` editing just two to three files and less than a hundred lines per round.
On the other end, `Claude Sonnet 4.5` or `GPT-5` generally make larger changes, with a much longer tail of sizable modifications.
We observe that this long tail typically comes from when models initialize test suites, create multiple versions of a submission to test against one another, or record insights as markdown notes to take forward into the next round.
We include two additional similar line charts that show the size of edits for the `README_agent.md` file along with any game-playing related core functionality.
The `Claude Sonnet 4` and `Claude Sonnet 4.5` models are relatively more extensive in their documentation.
`GPT-5` and `GPT-5-mini` exhibit a trend, where they take more notes up front, with a gradual decline into later rounds.
The remaining models do not fluctuate significantly in the amount of notes they take, with `o3` averaging under 10 lines changed per round.
Model changes to competition logic generally trends downward across rounds -- we generally observe that models define the majority of competitive logic early on, with later rounds consisting mostly of smaller, more specific adjustments.

**Models differ in the number of steps taken.**
Turn budget consumption is markedly different between models, with the Anthropic models and `Qwen3-Coder` usually using 22 to 27 turns out of the 30 turn limit.
On the other end, `Gemini 2.5 Pro` and `GPT-5 mini` rarely exceed 15 turns.
Figures on steps per round suggest that the number of steps models take from round to round is fairly steady; we were not able to identify any meaningful discrepancies in steps taken between rounds.
To further clarify -- although we impose the $1 per-round cost limit, there are *zero* occurrences across all tournaments we run of a model's trajectory being automatically terminated due to models exceeding the cost limit budget.
In other words, this means that the cost limit trend lines also faithfully reflect when models decide for themselves to stop editing for the round.
The majority of rounds end with a model producing a thought and action akin to "I have made all the changes I think are necessary. I will now conclude this round [END action]".

**Models differ in thought length.** While most models respond with similarly long thought traces, `Gemini 2.5 Pro` responds with significantly longer explanations, at around 95 words per response.
On the other end, `o3` is much more terse, with just under 19 words per response.
However, `o3`'s brevity comes with a heavy asterisk, as OpenAI's API is configured to hide intermediate thinking tokens for the `o`-series reasoning models.
The actual token count is thus likely vastly underestimated.

**Models are quick to recover from errant actions.** As discussed in Section 5.2, errant actions is not a significant factor in model performance.
The vast majority of actions (≥90%) are well formed and execute successfully.
We find that stronger models have slightly lower error rates, with `Claude Sonnet 4` at just 10.11%, while `Qwen3 Coder` tops out at 16.32%.
No arena has a particularly high errant action rate.

Furthermore, we also answer how quickly models recover from errant actions.
Prior work has reported that a major error mode of existing models are "cascading" failures -- if a model issues an errant action, the likelihood that it recovers successfully from the mistake decreases with every subsequent action.
In the year since these works pointed out this phenomenon, we find that such breakdowns have diminished significantly in frequency and length.
We observe that following an errant action, the next action is successfully more than 80% of the time.
By the third step following an errant action, there are nearly zero occurrences of models continuing to struggle to generate a well formed action.
In summary, our analyses strongly suggest that model performance in CodeClash is neither hindered by the choice of agent framework, nor that models are not adept at operating on the command line.

### D.2 Additional Ablations

**Multi-player settings are far more variable in standings.**
As mentioned in our results and analyses section in the main paper, we showcase the ability to run multi-player (3+) tournaments in CodeClash, specifically with the Core War arena.
As shown in Table 2 (list_arenas), four additional arenas -- BattleSnake, Halite, Poker, and RoboCode -- all support more running tournaments with 2 players, though we do not run comprehensive experiments due to both cost limitations and the analytical complexity introduced by multi-way competition, which we believe is best left as future work.
Lead changes are much more frequent as there are more players.
Furthermore, winners occupy a much smaller share of the total points in the 6 player arena compared to the head-on setting.

**Transparent codebases enable investigations in how models leverage views into others' development processes.**
We elected to run tournaments for CodeClash's main results under the assumption that models cannot view opponents' code because such a setting is more reminiscent of real world settings, where human players develop their solutions independently and have the option to keep their codebase closed source.
Therefore, we investigate the effects of making players' codebases viewable by opponents specifically as an ablation.
The introduction of this mechanic is potentially interesting as it shifts CodeClash much closer towards being a perfect information game, where all players in a game have knowledge of all relevant information in the system, including other players' decisions.
The knowledge of opponents' moves is what distinguishes a perfect information game like chess from an imperfect information game like poker, where opponent private cards are not known by default.

As mentioned in the main results, we carry out this investigation specifically for the Halite arena with three models (`GPT-5`, `Claude 4.5 Sonnet`, `Gemini 2.5 Pro`).
We found that the rate at which a player checks its opponent codebase fluctuates across both models and the phase of the tournament.
`Claude 4.5 Sonnet` is near constant, checking in on its opponent's activity nearly every single round.
`Gemini 2.5 Pro` and `GPT-5` both exhibit a trend where the check rate dips somewhat in the middle of a tournament before re-surging in later rounds.

### D.3 Analyzing Trajectories Using LMs as a Judge

This section describes detailed observations about the agent trajectories that were obtained using a LM as a judge setup.

**Additional results on groundedness, hallucinations, and validation:**
Notably, models behave very differently across arenas.
For example, BattleSnake elicits very strong hallucinations from Claude Sonnet 4.5 (affecting up to 45% of rounds), and RoboCode shows a particularly low rate of edit validation across models.

The kinds of edits that models perform change between rounds. While the initial editing of models is feature-heavy, as the tournament progresses, a larger amount of smaller tweaks or fixes appears together with rounds in which no meaningful edit was made to the main player file.

What models spend their turn on shifts from early to late tournament: read operations increase as the tournament progresses. It is also apparent how different the number of actions spent on testing, analyzing, and running test matches is between models.

**Groundedness of edits and validation of edits — structured output schema:**
```python
class BigQuestionsModelResponseSchema(BaseModel):
    """Schema for structured output of the model."""

    edit_category: Literal["tweak", "fix", "feature", "change", "none"]
    edits_motivated_by_logs: bool
    edits_motivated_by_insights: bool
    edits_motivated_by_old_static_messages: bool
    edits_reverted_based_on_insights: bool
    edits_tested_with_simulations: bool
    edits_validated_with_unittests: bool
    improved_test_analysis_framework: bool
    reasoning: str
```

The model is prompted with the groundedness and validation system prompt (see Appendix D.4 below).
The model then receives actions and outputs of the entire trajectory, however all thoughts of the models (i.e., all outputs of the models that are not the executable bash command) are stripped.
This is to avoid sycophantic tendencies of the judging LM model.

**Hallucinations — structured output schema:**
```python
source_categories = [
    "log", "sourcecode", "docs",
    "execution_output.test", "execution_output.analysis", "none",
]

claim_categories = [
    "loss_reason", "win_reason", "game_results",
    "possible_improvement", "player_code_behavior",
    "performed_edits", "misc",
]

class Incident(BaseModel):
    step_index: int
    claim_category: Literal[*claim_categories]
    claim: str
    source_category: Literal[*source_categories]
    source: str
    detailed_reasoning: str

class HallucinationResponseSchema(BaseModel):
    items: list[Incident]
```

**Action space analysis — structured output schema:**
```python
# Base categories
_read_subcategories = ["source", "logs", "docs", "other"]
_read_subsubcategories = ["new", "old"]

_write_subcategories = [
    "docs", "source.main", "source.main.backup",
    "source.opponent", "source.analysis", "source.tests", "other",
]
_write_subsubcategories = ["create", "modify_old", "modify_new"]

_execute_subcategories = ["game", "game.setup", "analysis", "unittest", "other"]
_execute_subsubcategories = ["in_mem", "new", "old"]

_all_categories = (
    ["search", "navigate", "submit", "other"]
    + [f"read.{sub}.{subsub}" for sub in _read_subcategories for subsub in _read_subsubcategories]
    + [f"write.{sub}.{subsub}" for sub in _write_subcategories for subsub in _write_subsubcategories]
    + [f"execute.{sub}.{subsub}" for sub in _execute_subcategories for subsub in _execute_subsubcategories]
)

class ActionCategoryResponse(BaseModel):
    category: Literal[*_all_categories]
    base_action: str
    success: bool
    notes: str = ""
    target_paths: list[str] = []

class ActionCategoriesModelResponse(BaseModel):
    categories: list[ActionCategoryResponse]
```

**Claude Sonnet 4.5 loses to a static solution written by a human expert.**
As discussed in Section 4.1 (Ablations), we run 10 tournaments of `Claude Sonnet 4.5`, the top model on the RobotRumble arena, against the top open-source submission we found on RobotRumble's online leaderboard (`gigachad` by `entropicdrifter`).

Additional details beyond the main paper:
- The top open source submission we use is ranked fourth overall (1554 Elo) on the leaderboard. Three additional, closed source submissions rank above, with the top submission ranking nearly 700 Elo points higher.
- While our main RobotRumble results ask models to write their bots in JavaScript, since the human submission is implemented in Python, for fairness, we ask `Claude Sonnet 4.5` to implement its bot in Python as well.

### D.4 Additional Analyses

**Models codebases are highly diverse, even when playing against the same opponent in the same arena.**
Continuing our discussion in Section 5.1 (Competitive Dynamics), we provide additional visualizations demonstrating how codebases evolve over time.
In round 1, model solutions are already quite divergent.
`Claude Sonnet 4.5` and `o3` tend to start off similarly, with the highest round 1 scores of 0.566 and 0.626 respectively.
The opponent doesn't seem to have too much of an impact on how similarly a model starts a tournament.
By round 15, models' solutions are unalike across the board, with `GPT-5` still maintaining the trend of being most diverse in its solutions (0.409 in round 1 to 0.163 by round 15).
Affirming our original claim, we find that model solutions are creative, even when facing the same opponent in the same arena multiple times.

**Model codebases become increasingly disorganized with time.** Continuing our discussion from Section 5.1 (Competitive Dynamics):
A higher root level clutter ratio (`files created in root` / `files created`) suggests that models are not expending effort or commands to organize files into aptly named subdirectories.
A lower file reuse ratio (`file reused at least once again after being created` / `files created`) suggests that instead of building on prior scripts and generating re-runnable code, models are creating a lot of single use files.
In our framing, desirable coding practices correspond to the top left quadrant (high file reuse, low root level clutter), while undesirable behaviors are in the bottom right (low file reuse, high root level clutter).
5 of 8 models fall in the bottom right corner.
`Claude Sonnet 4.5` shows the highest root level ratio.

As discussed in the main results, we notice that codebases tend to follow this trend of creating single use analysis and testing files that are then rarely reused later on in a tournament.
While we do not explore mitigating such behavior with prompting, we purport that this result is still noteworthy.
Refactoring and sustaining a well organized codebase is not something that models organically aspire towards.
We believe that CodeClash can serve as a testbed for investigating how LM managed codebases morph over time and exploring whether interventions in the form of data or external rewards can encourage better practices.

Finally, with analysis of filename redundancy over rounds, we find that the number of redundantly named files climbs upwards at different rates across all models.
`Claude Sonnet 4.5` creates 13 files with the prefix "analyze_".
From manual inspection, we found that most of these implementations are doing the same thing, with only the log file path being different.
The same trend holds for the "check_" and "ROUND_" files.
Such redundancy points to obvious room for improvement.
Long running SWE-agents that iterate and reuse a core set of files rather than spamming the codebase with single use scripts should be the more desirable behavior in the vast majority of use cases.

**Future code arenas.** We're particularly excited about the prospect of building new code arenas.
Similar to how task-oriented software development benchmarks like SWE-bench have led to a myriad of follow ups, we believe CodeClash's flexible definition for a code arena can incorporate existing simulators or inspire new environments for areas such as but not limited to cybersecurity, healthcare, and city planning.

---

## Appendix D.5: LM Judge System Prompts

### System Prompt for Groundedness and Validation Study

**Overall setting:**
You are an expert at analyzing the behavior of LM agents.
You are given a trajectory of actions of an LM agent that is playing a game.
You are asked to answer a series of questions about the behavior of the agent.

We are interested in:
1. What motivated the edits
2. What steps were taken to validate the edits

All questions that are marked as boolean need to be answered with a boolean value. You cannot answer "unknown" or similar.

**Main player file:** You are investigating an LM agent that is playing a game. The main player file is the main file that constitutes the agent's submission, i.e., the file that governs the agent's behavior and logic for the next round of the game that is being played. Commonly, this is the file called `main.py`, `player.py`, `robot.js`, `warrior.red`, or all relevant files in the directory `robots/custom/`. Do not confuse the main player file with analysis files, or copies of previous versions of the main player file or other bots that the agent is creating for testing purposes.

**Final edits:** The final edits are the changes to a file after all actions. For example, an edit action that is reverted by another edit action is not part of the final edits.

**Q1 (edit_category, one of none, tweak, fix, feature, change):** Categorize the kind of final edits to the main player file. Categories: (1) none: No change in behavior. Only comments, documentation, refactoring was performed. (2) tweak: Logic is left unchanged, but we do change some parameters. (3) fix: Small, targeted change with the intent to fix broken behavior. (4) feature: Significant new behavior is added, mostly extending the existing code. (5) change: We significantly change the behavior by rewriting significant logic of the code. Notes: Only count the final edits to the main player file (any edits that are reverted are not counted). For this question, only the main player file is considered. Precedence if multiple categories might fit: none < tweak < fix < feature or change. Ignore comments, documentation, or refactorings that do not change behavior.

**Q2 (edits_motivated_by_logs, boolean):** Are the final edits to the main player file motivated by previous round's logs? Answer True if ALL of the following is true: (1) A failure mode can be inferred with the help of reading the logs or analysis scripts evaluating the logs. (2) The edit is directly related to this failure mode. The logs can be either from a game that the player simulates itself, or from the previous round, but it must be a meaningful game log. Examples of real failure modes: the snake that the player is controlling runs out of food; our bot runs against a wall; our code times out. Examples of non-failure modes: Player 1 won 99% of the rounds; Player 2 is better most of the time.

**Q3 (edits_motivated_by_insights):** Can the goal of the final edits to the main player file be motivated by any insights based on the output of previous actions? If you answered True to Q2, answer True here as well. However, you can also answer True here, if one or more of the following is true: (1) The player wrote a meaningful test that revealed a problem (or a way to improve) and then performed the corresponding edit; (2) The player wrote a meaningful analysis script that revealed a problem (or a way to improve) and then performed the corresponding edit; (3) The player ran some test games that revealed a problem (or a way to improve) and then performed the corresponding edit; (4) The player made some changes, and then ran test games against the previous version and verified that the changes improved the performance, i.e., had a higher win rate.

**Q4 (edits_motivated_by_old_static_messages):** Were the final edits to the main player file motivated by old static messages, i.e., messages that are (1) Old: Were not created during the trajectory. (2) Static: Are always shown and do not depend on any tests or analysis outcomes. A common case is generic notes in `README_agent.md` or similar documentation proposing ways to improve the bot in the next round.

**Q5 (edits_reverted_based_on_insights):** Were any edits on the main player file reverted based on tests or simulations? Answer True if any edits to the main player file were reverted based on one or more of the following: (1) Unit tests showed that the edits introduced issues; (2) Simulations showed that the edits introduced issues or had a lower win rate.

**Q6 (edits_tested_with_simulations):** Are the final edits to the main player file validated by playing the game? In order to answer True, a real game has to be played. If there is an opponent, the new version has to win (or have a good win rate). Notes: If the games failed to run, or showed that the new version was clearly worse than the previous version, answer False. If it was not verified who won the games, also answer False. Unit tests do NOT count as a simulated game. Special case: If no final edits to the main player file have been made, answer True.

**Q7 (edits_validated_with_unittests):** Are the FINAL edits to the MAIN PLAYER FILE covered by specific unittests that test the new or modified behavior? Answer True, if the unittests cover (some of) the new behavior. Notes: Running the game to get a win rate does not count as a unittest, because it does not specifically validate specific changes. Running unittests that are unrelated to the changes does not count either. If the tests did not run, or showed that the new version was broken, answer False. Special case: If there are no significant changes, answer True.

**Q8 (improved_test_analysis_framework):** Was the test or analysis framework significantly improved and the player of the next round has more tools to realistically improve the bot? Examples of significant improvements: An additional test was added to a test script or unittest framework; The analysis script was improved to look for a new behavior or failure mode; A script to help running simulated games and to parse the results. Examples of non-significant improvements: Static messages or comments are added to the test or analysis framework; Documentation of the tests or analysis scripts; Analysis or test scripts that are specific to the current round and are not expected to be useful for the next round.

**Output format:** Answer in the json format specified. The reasoning field should contain an explanation for your answer that explains your reasoning for each of the answers. Include general statements/observations first, then write down your reasoning for each of the answers as Q1: <reasoning> double linebreak Q2: <reasoning>, etc.

---

### System Prompt for Hallucination Study

**Overall setting:**
You are an expert at analyzing the behavior of LM agents.
You are given a trajectory of actions of an LM agent that is playing a game.
We are interested in so called "incidents", ungrounded or hallucinated outputs from the LM of the agent.
For example, the agent might say that it spotted an issue in a game log, even though the log does not contain any information about the issue described.

**Steps:** The agent proceeds in steps. All steps together are called a "trajectory." You will see a step index for each step in the trajectory. Every step consists of a thought, an action, and an output. The thought is the text output of the agent, describing observations, thoughts, reasons for taking actions, or other information. The action is the command that the agent wants to execute. The output is the output of executing the command.

**Information of the agent:** The agent processes information from its previous steps. Sources include: Game logs from previous rounds; Reasoning about source code; Information from the output of executing tests; Information from the output of executing analysis scripts; Documentation.

**What constitutes an incident?** For a step to constitute an incident, ALL of the following must be true: (1) The thought is not framed as a hypothesis, but rather as a statement of fact. (2) The statement of fact is concrete. (3) The statement of fact in the thought cannot be corroborated by the information that the agent has access to at step i. (4) The agent also cannot come to the conclusion by common sense knowledge and reasoning about the information that the agent has access to at step i. (5) The agent would have had the means of obtaining the information in principle (analyzing logs, reading source code, executing tests, etc.). (6) The incident is relevant to the overall trajectory and the objective of the agent, i.e., the final goal of the agent winning the game.

**Examples of thoughts that constitute incidents:** "There is the following bug in the code" (but we did not see any code, or the bug is not actually present); "The log shows that we lost game 6" (but we only saw games 1-5); "We lost game 7 because our robot collided with the wall" (but previous information only shows that we lost game 7, not why).

**Examples of thoughts that do NOT constitute incidents:** "We can improve the code by doing X" (we did see relevant code, and with good reasoning, we could come to the conclusion that X is a good improvement, even though we did not execute tests or analysis scripts to verify this); "My bot is working perfectly" (this is just a slightly overconfident statement, but not a concrete claim).

**Claim categories:** loss_reason, win_reason, game_results, possible_improvement, player_code_behavior, performed_edits, misc.

**Source categories:** log, docs, sourcecode, execution_output.test, execution_output.analysis, misc, none.

---

### System Prompt for Categorizing Agent Actions

You are helping to analyze the actions of a LM agent (summarily referred to as "trajectory").
For every action, you return a category as specified by the structured output specs.

**Search operations:** search (grep or similar commands that search through files), navigate (Commonly navigate through the file system and discover files; includes commands like `ls`, `cd`, `pwd`, `find`, `tree`, etc.)

**Read operations:** The model reads code, documentation, logs, or anything else. Commands include `ls`, `cat`, `head`, `tail`, etc. Categories: read.source, read.logs, read.docs, read.other. Subsubcategories: x.new (script created in this trajectory), x.old (script created before any action seen).

**Write operations:** The model modifies files. Common commands include `cat ... > file`, `sed`, etc. Subcategories: write.docs, write.source.main, write.source.main.backup, write.source.opponent, write.source.analysis, write.source.tests, write.other. Subsubcategories: x.modify_old, x.create, x.modify_new.

**Execution operations:** Executions are anything that executes source files. Categories: execute.game, execute.game.setup, execute.analysis, execute.unittest, execute.other. Subsubcategories: x.in_mem, x.new, x.old.

**Other:** submit (player issues MINI_SWE_AGENT_FINAL_OUTPUT or COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT to finish the run), other (infrequent; only for write targets that are clearly not compatible with any other category).

**Category Priorities:** In order of importance: execution > writing > reading. So if an action combines writing with execution, the category should be execution, etc.

**Base actions:** The part of the command that describes the action best, but without any arguments. E.g., for `cat file.txt`, the base action is `cat`. For `cd /path/to/dir && python script.py`, the base action is `python`.

**Important notes:** (1) You MUST categorize EVERY action. Do NOT skip any action. (2) Every action MUST be put into exactly one category. (3) Your category MUST be one of the list above. (4) If you are unsure, use the best match for the category.
