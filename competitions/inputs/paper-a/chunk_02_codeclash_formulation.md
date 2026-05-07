# Chunk 02 — CodeClash: Formulation, Technical Details, and Features

---

## 2. CodeClash

### 2.1 Formulation

CodeClash formalizes competitive coding as a tournament, where two or more players compete in a code arena for multiple rounds.
*Player* refers to an LM equipped with an Agent Computer Interface (ACI) or scaffold that enables it to interact with a codebase.
Each player maintains their own codebase for the entire tournament.
A *code arena* is any competition platform that takes in multiple codebases and executes them against one another, producing measurable outcomes about relative performance on a designated objective (e.g., eliminating opponents, acquiring resources, maximizing profit).

Each round proceeds in two phases.
In the *edit* phase, each player independently modifies their codebase using whatever strategies they deem appropriate within a fixed budget of turns.
During the *competition* phase, all codebases are compiled and executed within the code arena, where they interact and compete directly against each other.
The arena determines a winner (or declares a tie) based on the codebases' performance.

CodeClash's formulation makes several key design decisions.
*Codebase-as-memory*: players have no explicit memory of actions from previous rounds.
Their information is limited to whatever they chose to record in the codebase.
*Log-based feedback*: after each competition phase, the results and logs are copied into each player's codebase as the sole source of new information.
*Strategic opacity*: players cannot see each other's codebases, though we explore lifting this restriction in Section 4.2.

### 2.2 Technical Details

To implement a player, we use `mini-SWE-agent`, an agent computer interface (ACI) that enables an LM to interact with a codebase by issuing `bash` actions to a terminal.
Each turn, the LM generates a ReAct style response containing a thought (in natural language) and a `bash` action, then receives standard output from the terminal environment in return.
Next, we define a lightweight, flexible interface for a code arena.
An implementation only needs to define commands to run the competition and determine a winner.
This minimal overhead enables us to fold many existing competitive programming games and tasks into CodeClash.
More technical discussion in Appendix A (Infrastructure).

### 2.3 Features

CodeClash's initial release features a suite of 6 code arenas, as listed in Figure 1.
Each arena is covered thoroughly in Appendix B (Arenas).
CodeClash introduces several distinctive properties that collectively push models beyond traditional code completion and issue resolution.

**Open-ended objectives.** CodeClash departs from the traditional reliance on unit tests or implementation correctness to measure success.
Instead, players code to win competitive outcomes that vary dramatically across arenas, from maximizing profit to surviving the longest.
This mirrors the ultimate objectives of real-world software more faithfully, where code is written to achieve tangible, practical outcomes (e.g., maximize resources, generate revenue, outperform competitors) rather than simply achieving technical correctness.
A consequence of rich objectives is that models must then decompose a higher-order goal into actionable subtasks and measurable, intermediate metrics to inform code improvements.

**Diverse arenas.**
CodeClash's arenas vary significantly, with drastic differences in a codebase's structure, how a codebase interfaces with the arena engine, and the types of logs and feedback generated.
This contrasts sharply with existing benchmarks, where evaluation follows a consistent pattern of problem statement, code implementation, and test validation.

**Adversarial adaptation.** CodeClash's uniquely multi-player, head-to-head setting adds a new layer of complexity to coding evaluations.
While decent LMs may be capable of writing competent implementations, top-performing players will analyze opponent behaviors and incorporate countermeasures, all the while being indecipherable in their own play.
Early round wins do not ensure continued dominance.
At some point, the challenge shifts from writing good code to writing code that consistently beats intelligent competition.

**Self-crafted memory.** As mentioned in Section 2.1, CodeClash does not maintain persistent memory for models across rounds; only ephemeral, within-round memory exists.
To retain information for future use, models must explicitly add insights to the codebase; how to represent such knowledge is left entirely to the model's discretion.

**Self-directed improvement.** Beyond a brief description of the environment and arena, the initial system prompt provided to each player at the start of every edit phase contains *no* guidance beyond high level suggestions about how to enhance its codebase.
All decisions and changes LMs make are necessarily autonomous.
In practice, this may manifest as models writing analysis scripts to understand competition logs, maintaining notes about past rounds or opponents, or generating multiple candidates to test against one another.
