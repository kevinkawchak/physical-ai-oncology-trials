# Chunk 08 — Appendix A: Infrastructure and Appendix B: Arenas

---

## Appendix A: Infrastructure

In this section, we provide some additional insights and discussion into the tooling and infrastructure that CodeClash uses to (1) enable LMs to edit codebases and (2) automatically run codebases against each other within the code arena.
Mimicking Figure 1, we provide a more technically informative breakdown of the CodeClash loop in Figure (tech-overview).

**Figure tech-overview Caption:** Technical overview of a CodeClash round. Each round, during the *edit* phase, LMs edit their respective codebases within Docker containers, using `mini-SWE-agent` to facilitate multi-turn editing (Step 1). This is followed by the *competition* phase, where the codebases are copied the arena docker container (Step 2). The arena then runs codebases against each other, with the game-play and outcomes captured as logs (Step 3). These logs are copied into each player's codebase before the next round begins (Step 4).

We format our discussion of CodeClash's infrastructure as a series of system design questions that reflects the thought processes we went through and decisions we arrived upon towards implementing CodeClash.

**How should models edit their codebases?** The benefits and drawbacks around methods for how LMs interact with codebases has been investigated thoroughly by recent works.
Inspired by both prior research insights and current, popular paradigms for AI coding tools, we wanted to ensure several key properties for how LMs should manipulate a codebase for CodeClash, which is step 1 in Figure tech-overview.

1. LMs should be able to *view execution feedback*. Execution is crucial to enable models to create and use their own constructs (e.g., analysis scripts, memory systems).
2. LMs should be able to *interact with a codebase*. A defining challenge of CodeClash is that LMs operate in a self-directed manner. Workflow-oriented approaches are unsuitable for our setting. Going hand-in-hand with (1), interaction is also necessary so that models can string sequences of changes together.
3. LMs should *operate using `bash` actions, not tools*. As described in prior work, various workflows and tools can be (un-)intentionally biased to favor particular models. Our goal is to evaluate models, not scaffolds or tools. Therefore, we decide to make LMs operate in the most "impartial" action space. This decision also leaves an opportunity for LMs to synthesize their own tools across rounds.

Considering these points all together, we found `mini-SWE-agent` to be most suitable.
`mini-SWE-agent` is a lightweight agent scaffold that allows LMs to interact with a codebase in a terminal environment.
Per turn, an LM generates a `bash` command, then receives standard output as execution output.
The combination of `mini-SWE-agent` and Claude 4 Opus scores 67.6% on SWE-bench Verified, giving us confidence that the models we evaluate are capable of performing bash-only interactions with a low to non-existent rate of failures due to syntactic errors such as malformed responses or actions.

**How do we make CodeClash portable and reproducible?** Following precedent established by existing interactive coding benchmarks, we use Docker to containerize the environments for (1) LMs to develop their respective codebases (*agent containers*) and (2) running codebases in the arena (*arena container*).
No codebase edits or arena runs are ever performed on device.
The only artifact created on the local machine are logs capturing tournament metadata and outcomes.

**What initial assets should a model be given?** In other words, what should the starter codebase specific to each arena generally contain?
To answer this, we outlined a shortlist of several behaviors and conditions that should be supported and true for any arena.

- LMs should be able to learn about the arena/game as extensively as it would like. We do not assume players have any prior knowledge about how the arena works.
- LMs should be able to run the arena to understand it and perform testing.
- LMs are provided with a simple but functional baseline strategy that demonstrates core mechanics. A player does not need to code a valid submission from scratch.

Based on this, we make sure every codebase has the following assets:

- *Documentation*: For every arena, we were able to find source code containing arena documentation (e.g., https://github.com/BattlesnakeOfficial/docs). We copy documentation into a `docs/` folder for every arena's starter codebase.
- *Arena executable*: Any executables and assets needed to run a round of the arena are fully available to each player. However, the exact `bash` commands are not disclosed; the burden remains on the model to figure out how to use assets.
- *Working submission*: Like how human participants are provided a simple, functional, and suboptimal baseline strategy, LMs are given a starter codebase that can be submitted as is. This ensures meaningful competition from the first round.

In practice, for any arena, the starter codebases for each player and the codebase for running the competition across multiple codebases are identical.

**Per round, how many times should a competition be run?**
This question stems from the non-determinism that we observed in the majority of CodeClash arenas.
With the exception of MIT Battlecode 2025, we found that given the same codebases and the same arena, the outcome of a single simulation is indeterminate, which is to be expected.

In order to declare a winner with confidence, each round at step 3 in Figure tech-overview, the arena runs the competition 1000 times.
We declare the winner as whichever player wins the most out of the 1000 simulations (or declare a tie if ties are most frequent), rather than requiring a specific win percentage threshold.
This approach aligns with standard practice in competitive gaming communities and avoids introducing arbitrary performance cutoffs.
We concretely review how we calculate win rate and Elo in Appendix C.3 (Evaluation Metrics).

**How can models improve their codebase?** A cornerstone to performing well in CodeClash is a model's ability to understand past rounds' outcomes, then adapt the codebase to perform better in the arena against the opponent(s).

To encourage such behavior, both the proceedings and outcome of each simulation are logged.
The precise format of the logs depends on the arena.
These logs are then copied from the arena container back into the agent containers, specifically in a designated `logs/` folder within the agent's codebase, as reflected by step 4 in Figure tech-overview.

How the model interprets these logs or acts upon them is entirely self-driven.
In the initial system prompt, we generally mention that analyzing logs might be helpful, but we do not provide any arena-specific advice on how exactly logs should be interpreted.
In practice, we've observed a spectrum of interesting approaches.
Models will directly read the raw logs, write scripts to solicit insights, or even modify the logs.
More insights in Appendix D (Extended Results).

**What happens if a model's codebase is not a valid submission?** We observed during early trials that models will occasionally errantly modify a codebase such that it no longer functions properly when run in the arena.
The error modes are most frequently due to certain expectations about the codebase not holding. For instance...

- For Battlecode, the main bot logic should be represented entirely within a `./bot.py` file that implements a `turn` function.
- For Battlesnake, the bot is in `main.py`, which implements a `move` function.
- For RoboCode, the tank bot should be defined under `robots/custom/`, and the code must pass compilation (`javac -cp "libs/robocode.jar" robots/custom/*.java`).

We note that we do not define these constraints -- these rules are reflective of the original conditions these arenas and games impose on human players and their submissions.

To address this, we first, implement per-arena validation to check that the codebase is ready for competition.
The check is run at the outset of step 3 in Figure tech-overview.
Second, we define the following decision tree to handle situations where 1+ players have invalid codebases.

- If all player codebases are invalid, the round is declared a tie.
- If only one player codebase is valid, that player is declared a winner.
- If 2+ player codebases are valid, the competition phase is run with all valid codebases. Any invalid codebases are excluded.

**Do arenas have positional advantages, and how are such advantages accounted for?**
A *positional advantage* refers to a situation where, assuming 2+ players have identical codebases, one player consistently wins.
We want to eliminate such advantages in CodeClash, as they unfairly affect the arena outcome in ways that are outside of a player's control.

To detect whether positional advantages are present in an arena, we run the aforementioned experiment -- for every arena, we run a tournament with two "dummy" players that do not change the initial codebase.
Each tournament is run for 25 rounds, and the order of players is fixed.
We then check round outcomes, with the expectation that ~50% win rate suggests no such positional advantages are present.
From this investigation, we found MIT Battlecode 2025 to be the only arena that showed evidence of positional advantage.

However, checking for positional advantages may be tedious to repeat constantly for new arenas or when arena settings are adjusted (e.g., the `map` being used for Battlecode, `battleField` dimensions for RoboCode).
Therefore, to reliably eliminate any advantage, we simply randomly shuffle the order of players with equal probability at step 3 in Figure tech-overview, immediately after the codebase validation step.
We verified this fix by re-running the prior experiment for MIT Battlecode 2025 and found that the win rate returned back to 50%.

**Trajectories are tedious to parse.** Reading arena logs and `mini-SWE-agent` editing trajectories in their raw form was extremely laborious.
To make it easier to understand what has happened throughout the course of a tournament, we wrote a viewer for CodeClash logs that provides friendly visualizations of log content and automatically calculates some game statistics (e.g., p-value calculation to indicate if a round winner is statistically significant).

---

## Appendix B: Arenas

This section contains arena cards describing each of code arena supported in CodeClash.
Per arena, we cover the objective(s), arena mechanics, log formats, and effective strategies.
We summarize all arenas supported in CodeClash in Table 2 (list_arenas).

### B.1 MIT Battlecode 2025

The MIT Battlecode organization is a student run group at the Massachusetts Institute of Technology that creates and hosts coding competitions.
CodeClash specifically supports the 2025 edition of the competition.
As described on the website (https://battlecode.org/):

> Battlecode is a real-time strategy game in which you will write code for an autonomous player. Your player will need to strategically manage a robot army and control how your robots work together to defeat the enemy team.

**System Prompt Description of Battlecode:** Battlecode 2025 throws you into a real-time strategy showdown where your Python bot pilots a team of specialized robots—Soldiers, Moppers, Splashers—alongside towers that spawn units or generate resources. Your mission: paint over 70% of the map (or eliminate the enemy) by coordinating cleanups, area cover, and tower-building through tight bytecode budgets and clever unit synergy.

**What are effective strategies?**
Some effective approaches include efficient algorithms for path-finding/exploration, coordinating communication between agents, and finding the right balance between offensive moves (e.g., attacking, painting, destroying towers) and defensive measures (protect territory, tower placement, maintain stream of resources).

**What assets are provided in the initial codebase?** `run.py/` is the python script used to run players and upgrade versions. `src/` is the directory meant to contain all player source code and, `test/` contains all player test code. `client/` contains the client and the proper executable can be found in this folder. `matches/` is the output folder for match files. `maps/` is the default folder for custom maps.

**What are the arena configurations?**
For the 2025 edition "Chromatic Conflict", two teams of virtual robots roam the screen, managing resources and executing different offensive strategies against each other. Two types of resources exist in the arena: Money and Paint. Money is needed to produce units, buy towers and activate economy boost patterns (called SRPs). Paint is needed to produce units, for the win condition, to resupply units with paint and to paint special patterns, which were prerequisites for acquiring SRPs and towers. There are also two kinds of soldiers: Moppers and Splashers. Moppers can attack other units without costing paint, which makes them the only unit capable of surviving indefinitely without a tower. They can also clean up enemy paint, making them essential for cleaning up enemy paint off of ally patterns. Splashers can paint over enemy paint with ally paint and are the only unit which can paint several squares at once. The last component of the arena is towers which are immobile units that can spawn units. Money and Paint Towers will passively generate the corresponding resources. Defense Towers have high damage output and generates chips upon attacking enemy units.

**How is the winner determined?**
The winner is the first team that is able to "paint" 70% of the map.

**How are arena logs formatted?**
The arena logs are written as a sequential record of the match.
They begin with setup information, including which bots are playing and on which map.
After that, each line corresponds to a turn, tagged with the acting player and unit, followed by the action taken (e.g., spawning a new robot, attempting to build a tower, or performing a mop swing attack).
In effect, the log provides a turn-by-turn narrative: what units were created, what abilities were triggered, and how each side attempted to advance.

**Example of BattleCode Log:**
```
Playing game between p1 and p2 on quack
[server] -------------------- Match Starting --------------------
[server] p1 vs. p2 on quack.map25
[A: #1@1] BUILT A MOPPER
[B: #4@1] BUILT A MOPPER
[A: #1@2] BUILT A SOLDIER
[B: #2@2] BUILT A SOLDIER
[A: #3@2] BUILT A MOPPER
[A: #12138@3] Trying to build a tower at (18, 25)
[B: #13376@3] Trying to build a tower at (18, 9)
[B: #4@4] BUILT A MOPPER
[A: #12523@4] Mop Swing! Booyah!
```

---

### B.2 Battlesnake

Battlesnake is a multi-player game, where each player's code controls a snake operating on a grid.
The arena's rules and objectives are heavily reminiscent of the traditional snake game.
The general objective is to program your snake to survive as long as possible.

The game starts with 2+ snakes positioned at different quadrants of the grid.
Throughout the course of the game, food pellets will pop up -- if a snake consumes (moves into a cell containing) a pellet, the snake's body gets longer by one cell.
There are several ways a snake can "die".
If it collides with a wall, its own body, or another snake that is longer, the snake is eliminated.
If the snake does not make a legal move on any particular turn, the game also ends.
The winner is the last remaining snake, or the longest snake if multiple are alive upon the exhaustion of some turn limit.

**System Prompt Description of Battlesnake:** You are a software developer ({{player_id}}) competing in a coding game called Battlesnake. Your bot (`main.py`) controls a snake on a grid-based board. Snakes collect food, avoid collisions, and try to outlast their opponents.

**What are effective strategies?** Effective Battlesnake bots rely on strategies that balance safety, space control, and efficient movement.
A common approach is to use *flood-fill or area estimation* to avoid moves that lead into regions with insufficient space, reducing the chance of being trapped.
*Pathfinding algorithms such as A** help snakes reach food or navigate safely around hazards, often incorporating penalties for risky tiles near enemy heads.
Many bots also implement *look-ahead search*, simulating several future turns to predict collisions and maintain advantageous positioning.
Finally, strong bots prioritize *risk-aware heuristics*, such as only engaging opponents when longer or only pursuing food when health is low.

**What assets are provided in the initial codebase?**
The `docs/` folder serves as the full documentation hub for the Battlesnake platform, containing subdirectories such as `api/`, `guides/`, maps/, and policies/, which collectively explain how to use the Battlesnake API, configure maps, follow gameplay policies, and get started with development. It also includes Markdown files like README.md, index.md, and quickstart.md for setup instructions; rules.md detailing official game rules and snake behavior; faq.md answering common developer questions; and starter-projects.md offering templates for new Battlesnake projects. Complementing the documentation, the game/ directory contains the full Go implementation of Battlesnake's core logic. Key source files such as board.go, ruleset.go, standard.go, and pipeline.go define how the game board is represented, how rules are enforced, and how turns are processed. Specialized variants of the game board like royale.go, solo.go, constrictor.go, and wrapped.go implement different modes. Other files in the root directory include main.py, which serves as a starter template for Battlesnake logic and helper functions, server.py for server setup and request handling, requirements.txt listing Python dependencies, and a Dockerfile for containerized deployment.

**What are the arena configurations?** The Standard Arena in Battlesnake is the default game environment, adhering to the core game rules without any modifications. In this arena, the number of Battlesnakes can vary, ranging from a 1v1 match or multiple snakes competing, such as four or eight. The game board is a square grid measuring 11×11 cells, totaling 121 cells. Each cell is a discrete unit where snakes and food can occupy. The arena's boundaries are defined by the edges of this grid, and snakes are restricted to moving within these confines. Movement is allowed in four directions: up, down, left, and right, with no diagonal movement permitted. At the start of the game, snakes are placed at random positions within the arena, and food items are similarly distributed across the grid.

**How is the winner determined?** In Battlesnake, the winner is determined by being the last remaining snake on the game board. Each snake takes turns moving, loses one health point per turn, and can regain health by consuming food, which also causes the snake to grow in length. Snakes are eliminated in several ways: colliding with their own body, colliding with another snake's body, or engaging in a head-to-head collision with another snake. In head-to-head collisions, the longer snake survives while the shorter one is eliminated. If both snakes are the same length, both are removed from the game. Players must carefully manage their health, navigate the board without running into obstacles or other snakes, and strategically consume food to survive longer than their opponents. The game continues until only one snake remains, and that snake is declared the winner.

**How are arena logs formatted?** The log for a single competition run is represented as a single `.jsonl` file, where each line in the file is a dictionary corresponding to a single turn of the run.
Each line of a Battlesnake log records the complete state of the game at a given turn.
It captures the ruleset and configuration, the current turn number, the map dimensions, and the positions and attributes of all snakes (their ID, health, body coordinates, head position, and length).
It also lists the placement of food and hazards at that moment, as well as the perspective of the specific snake whose API is being called.
In other words, every log entry is a snapshot of the board state.

**Example of BattleSnake Log:**
```json
"turn": 0,
"board": {
"height": 11,
"width": 11,
"snakes": [
  {
    "id": "794bb7d7-a1ee-4939-a664-dd77d3c5f6e3",
    "name": "p1",
    "latency": "0",
    "health": 100,
    "body": [{"x": 9, "y": 9}, {"x": 9, "y": 9}, {"x": 9, "y": 9}],
    "head": {"x": 9, "y": 9},
    "length": 3,
    "shout": "",
    "squad": "",
    "customizations": "color": "#888888", "head": "default", "tail": "default"
  }
```

---

### B.3 Core War

For Core War, players write small assembly-esque programs (called a "warrior").
The programs are run in a simulated, shared virtual memory.
The goal of every program is to disable all opposing programs.
The ultimate objective is to be the last program standing.

A unique facet of Core War is that the programming language, RedCode, is specific to the game.
RedCode supports basic operations (e.g., `mov`, `add`, `jump`, `compare`) along with multiple addressing modes (e.g., immediate, direct, indirect).
Warriors compete in the "core", which generally is a fixed size, circular memory array that resembles main memory (RAM).
The core is represented by a simulator called MARS.
The execution of the game then proceeds in cycles, where each cycle, the simulator alternates between warriors and executes on instruction per active process.
If a process executes an invalid instruction or hits an illegal condition, the process dies.
Warriors can also be designed to spawn additional processes with special instructions (`SPL`).
If all of a warrior's processes are killed, it is eliminated.
Core War games are typically played a maximum number of cycles; if no warrior is eliminated by the end, the round is a draw.

**System Prompt Description of Core War:** You are a software developer ({{player_id}}) competing in a coding game called Core War. Core War is a programming battle where you write "warriors" in an assembly-like language called Redcode to compete within a virtual machine (MARS), aiming to eliminate your rivals by making their code self-terminate. Victory comes from crafting clever tactics -- replicators, scanners, bombers -- that exploit memory layout and instruction timing to control the core.

**What are effective strategies?**
Core War warriors typically incorporate three dimensions -- offense, defense, and adaptability.
A common offensive strategy is to write loops that scatter "bombs" (invalid instructions) into memory, similar to the program in Figure arena:corewar:snippet.
Another approach is to write programs that replicate as much as possible to increase survival rate.
An advanced warrior will usually combine such tactics.

**What assets are provided in the initial codebase?** The codebase contains three main directories `config/`, `docs/`, and `src/` and provides a complete Core War environment, including the assembler, simulator (virtual machine), documentation, and example warriors. In `config/`, different files define different configuration profiles for the pMARS simulator, allowing tournaments or simulations under multiple rule sets and tuning the VM for different "arena sizes." The `docs/` folder describes how Core War works and how to write Redcode warriors. `src/` provides source code for the pMARS simulator and assembler, including files that implement the display and UI modules, core files, and configuration.

**What are the arena configurations?** Core War is a game in which two or more virus-like programs fight against each other in a simulated memory space or core. Core War programs are written in an assembly language called Redcode which is interpreted by a Core War simulator or MARS (Memory Array Redcode Simulator). The object of the game is to prevent the other program(s) from executing. At the start of a match, each warrior is loaded into a random memory location. Programs take turns executing one instruction at a time. A program wins by terminating all opponents, typically by causing them to execute invalid instructions, leaving the victorious program in sole possession of the machine.

**How is the winner determined?** In the standard Core War rules, the winner is determined by being the last warrior still "alive" (i.e., having at least one process still running) or the last to execute a valid "live" instruction. A warrior "dies" when it has no remaining processes left. Processes can die if they execute an invalid instruction or are overwritten.

**How are arena logs formatted?**
Core War logs generally report the outcomes, like which warrior survived, how many "processes" (active execution threads) they maintained, or how many cycles elapsed before the match ended.
These logs don't usually show step-by-step instruction execution, but instead give you a high-level summary of win/loss/tie, survival, and match duration.

**Example of Core War Log:**
```
Program "Dwarf" (length 4) by "A. K. Dewdney"

       ORG      START
START  ADD.AB #     4, $     3
       MOV.I  $     2, @     2
       JMP.B  $    -2, $     0
       DAT.F  #     0, #     0

Dwarf by A. K. Dewdney scores 3
Dwarf by A. K. Dewdney scores 0
Results: 1 0 0
```

---

### B.4 Halite I

For Halite, players write autonomous bots that battle head to head with the goal of taking over the largest share of a virtual grid. Each bot issues commands every turn to move, collect, and deposit halite — a valuable in-game resource. The objective is to maximize your halite by the end of the match while strategically navigating around opponents and avoiding collisions. Bots use their strength to gain territory, and their territory to gain strength—outmaneuvering opponents based on the relative sophistication of their code.

A distinctive aspect of Halite is that it combines algorithmic strategy with real-time resource optimization. Players can program their bots in one of 4 languages (C, C++, OCaml, and Rust), and the game environment simulates simultaneous turns, where every decision — from choosing optimal collection routes to predicting enemy movements — can make the difference between victory and defeat. Matches are visualized in an animated replay, saved as an `.hlt` file, allowing players to analyze and refine their bot's performance across different maps and opponents.

The Halite series also includes Halite II and Halite III, follow up iterations to the initial competition with significant updates to the nature of the competition.
We doubly clarify that this version of Halite described here refers specifically to Halite *I*, released in 2016.
We are planning to support Halite II and Halite III in CodeClash in the near future.

**System Prompt Description of Halite:** Halite is a multi-player turn-based strategy game where bots compete on a rectangular grid to capture territory and accumulate strength. Players control pieces that can move across the map to conquer neutral and enemy territory, with each cell providing production that increases the strength of pieces occupying it. The goal is to control the most territory by the end of the game through strategic expansion, consolidation of forces, and tactical combat decisions. You have the choice of writing your Halite bot in one of four programming languages: C, C++, OCaml, or Rust. Example implementations can be found under the `airesources/` folder. Your submission should be stored in the `submission/` folder.

**What are effective strategies?**
Effective strategies in Halite span three distinct phases. During the early game up until the bot makes contact with an opponent, an effective strategy is to capture neutral territory to fuel your growth with production and deprive other players of valuable neutral territory. Since bots don't yet have to defend their territory from other players, quick expansion into the most valuable areas is vital. During the mid-game (from when bots first make contact with another bot until there is very little remaining valuable neutral territory), players may want to shift to a hybrid of defense and offense: protect the best regions, seize remaining valuable neutral territory, and begin targeting weak points of opponents. Then, during late game, with most neutral territory gone, the game becomes purely about taking territory from other players. Players that take advantage of overkill and attack enemies' high production areas are more likely to win.

**What assets are provided in the initial codebase?**
The initial Halite codebase provides all the foundational tools a player needs to create and test a functioning bot. Each starter package includes template code for your bot, such as a MyBot file where you implement decision-making logic, along with helper libraries that handle communication with the game environment (for example, receiving map data and sending moves). It also comes with a "RandomBot" or simple baseline bot to use as a reference, plus utilities for local simulation and visualization so you can test games without uploading them. These assets are designed to let players quickly get started with writing a bot that reads the game state, decides on moves, and interacts with the game engine via the provided API.

**What are the arena configurations?**
Halite games take place on a two-dimensional, rectangular grid map whose width and height are randomly generated for each match. The exact dimensions vary, but the generator always ensures that the resulting map is symmetric—it creates one section, then tessellates, reflects, and shifts it to fill the full board. This symmetry guarantees fair starting conditions for all players. Each cell on the map has two key values: Production, which determines how much Strength a stationary piece gains each turn, and Strength, representing how powerful a piece currently is. The maps are designed to be "interesting," with clusters of high- and low-production zones rather than random noise, encouraging strategic territorial expansion. The map wraps around at the edges, meaning that moving off one side (for example, going North from the top row) places a piece on the opposite edge of the map—making the grid behave like a torus. The coordinate origin (0,0) is located at the northwest (top-left) corner of the map.

**How is the winner determined?**
Halite is played on a rectangular grid. Players own pieces on this grid.
Some pieces are unowned and so belong to the map until claimed by players. Each piece has a strength value associated with it.
At each turn, bots decide how to move the pieces they own. Valid moves are: STILL, NORTH, EAST, SOUTH, WEST.
When a piece remains STILL, its strength is increased by the production value of the site it is on. When a piece moves, it leaves behind a piece with the same owner and a strength of zero.
When two or more pieces from the same player try to occupy the same site, the resultant piece gets the sum of their strengths (this strength is capped at 255).
When pieces with different owners move onto the same site or cardinally adjacent sites, the pieces are forced to fight, and each piece loses strength equal to the strength of its opponent.
When a player's piece moves onto an unowned site, that piece and the unowned piece fight, and each piece loses strength equal to the strength of its opponent.
When a piece loses all of its strength, it dies and is removed from the grid.
The game ends when only one player remains, or when a maximum number of turns has elapsed, defined as 10×sqrt(width × height).
If the turn limit is reached or multiple bots are eliminated simultaneously, players are ranked by the amount of territory they control, with total Strength acting as a rare tiebreaker.

**How are arena logs formatted?**
Arena logs in Halite are formatted as sequential text entries that record the setup, turns, and results of a match. The log typically begins with the paths to the submitted bot executables for each player, followed by the map size or configuration, and then messages confirming initialization for each bot. Each turn of the game is listed sequentially (e.g., Turn 1, Turn 2, ...), representing the progression of the match. At the end, additional metadata is provided, such as the map seed, the path to the replay file, and final rankings with information about which bot lasted the longest. This structured format allows both human review and automated parsing to analyze bot performance.

**Example of Halite Logs:**
```
/p1/submission/main.o
/p1/submission/main.o
/p2/submission/main.o
/p2/submission/main.o
34 34
Init Message sent to player 2.
Init Message sent to player 1.
Init Message received from player 1, MyCBot.
Init Message received from player 2, MyCBot.
Turn 1
Turn 2
...
Map seed was 4244905440
Opening a file at /logs/1761005260-4244905440.hlt
Player #1, MyCBot, came in rank #2 and was last alive on frame #340!
Player #2, MyCBot, came in rank #1 and was last alive on frame #340!
```

---

### B.5 Poker (Husky Hold'em Bench)

Using the Husky Hold'em Bench poker engine, CodeClash supports the standard, No-Limit Texas Hold'em style of poker.
As a refresher, each player gets two private cards.
Five community cards are revealed across four stages, and players bet freely (maximum of stack size) to win chips by making opponents fold or making the best five-card hand.

The poker engine deals blinds (small/big), then runs usual betting rounds -- pre-flop, flop, turn, river -- and enforces the turn order, legal actions (check/call/raise/fold), and pot accounting.
As mentioned, the rules are explicitly *no-limit*, so bets are variable size.
The design of the engine makes implementation of a poker bot straightforward.
A player client simply has to choose actions via a simple interface that lists the valid actions.

**Isn't poker solved already?** Poker has served as a long standing sandbox for researching superhuman level AI systems.
Simple, constrained variants of poker, such as Heads-Up [No-]Limit Texas Hold'em (2 players, fixed bet sizes) have effectively been solved or close to solved by systems such as Cepheus, Libratus, and Pluribus.
However, multi-player settings with three or more participants (in other words, *not* Heads-Up, player versus player) are far from solved, as complexity skyrockets with more players.

**What are effective strategies?** We briefly outline several well-established principles that contribute to the design of strong poker bots, while noting that this overview is not exhaustive given the depth of prior research.
Effective agents often rely on game-theoretic strategies to approximate equilibrium play, ensuring they are difficult to exploit over long horizons.
At the same time, they incorporate opponent modeling and randomization to adapt to behavioral patterns while remaining unpredictable, and use bet-sizing heuristics to balance pressure against risk in pursuit of long-term expected value.

**What assets are provided in the initial codebase?**
The initial codebase includes a full stack for a poker application: the `engine/` directory contains the core game logic and simulation framework (deck, hand-evaluation, betting rounds, rules, player abstractions, and state transitions), while the `client/` directory implements the user interface, sample clients or bots, configuration files (e.g., for game parameters such as blinds, player stacks, seating), and documentation/support files. Together, the codebase provides everything needed to run poker matches, build or plug in client agents or user interfaces, configure game variants, and execute games or simulations.

**What are the arena configurations?**
The arena in this context represents the virtual poker table managed by the `pokerden-engine`. Configuration settings define parameters such as the number of seats (players per table), initial chip stacks, blind levels (small and big blinds), betting structure (limit, no-limit, or pot-limit), deck configuration, and game type (e.g., Texas Hold'em, Omaha). These parameters are typically specified in configuration or initialization files that the engine reads at startup, ensuring all clients connect to a consistent game environment. The engine controls turn order, manages rounds (pre-flop, flop, turn, river), and enforces timing or betting limits. In tournament or simulation setups, multiple tables (arenas) may run concurrently with identical rule configurations but independent game states.

**How is the winner determined?**
Within each hand, the `pokerden-engine` determines the winner by evaluating all active players' final hands at showdown using standard poker hand rankings—from high card up to royal flush. If a player causes all others to fold, that player automatically wins the pot without showdown. At showdown, the engine compares hand strengths computed through its hand evaluation module, distributing the pot accordingly (splitting it in case of ties). Over a series of hands or a full match, the overall winner is the player (or client agent) with the largest remaining chip count when the game ends—either after a fixed number of rounds, when all but one player has been eliminated (tournament mode), or when the match duration concludes (cash-game simulation).

**How are arena logs formatted?**
The poker logs record each hand as a sequence of betting rounds, listing player actions (e.g., raise, call, check) along with bet sizes, updated pot totals, and any side pots.
They also include the community board cards, each player's hole cards, and timing information for decisions.
At the end of the hand, the logs report chip deltas and final balances, providing both a detailed play-by-play and a clear summary of outcomes.

**Example of Poker Log:**
```json
  "gameId": "8ee11ef4-ffcb-4c42-8ccf-7865a94a3ae5",
  "rounds": {
    "0": {
      "pot": 15,
      "bets": {
        "982465989": 5,
        "3161785489": 10
      },
      "actions": {
        "982465989": "RAISE",
        "3161785489": "RAISE"
      },
      "action_sequence": [
        {
          "player": 982465989,
          "action": "RAISE",
          "amount": 5,
          "timestamp": 1761005394049,
          "pot_after_action": 5,
          "side_pots_after_action": [
            {"amount": 5, "eligible_players": [3161785490, 982465990]}
          ],
          "total_pot_after_action": 5,
          "total_side_pots_after_action": [
            {"id": 0, "amount": 5, "eligible_players": [3161785490, 982465990]}
          ]
```

---

### B.6 RoboCode

RoboCode is a 2+ player game where your code represents a tank in a 2D grid battlefield.
The ultimate objective is to outlast and outscore opposing tanks.

Each tank has a set of actions -- your tank can move around, turn (body, turret, radar), detect other bots, and fire bullets.
There are several factors to take into account when encoding strategy.
First, in addition to a health bar, each tank also has an energy bar that is expended when firing, so players have to be mindful about spamming shooting.
Second, bullets take time to travel, so shots should be directed towards anticipated positions of opposing tanks.
A match continues until only one tank remains standing or the round limit is reached, with scores awarded for survival, damage dealt, and final placement.

**System Prompt Description of RoboCode:** You are a software developer ({{player_id}}) competing in a coding game called RoboCode. Robocode (Tank Royale) is a programming game where your code is the tank: each turn your bot sends intents—speed plus body/gun/radar turn rates and firepower—based on the game state it perceives via radar. Your program decides how to move, aim, and fire in a deterministic, turn-based arena to outlast other bots.

**What are effective strategies?** A key theme to successfully RoboCode bots is *predictive targeting* -- where your tank fires should account for estimations of opponents' future locations, based on their speed and direction.
*Wave surfing* refers to a tactic that assumes opponents' bullets will be directed in a way that mimics "expanding waves"; movement patterns attempt to minimize the chance of being hit under this assumption.
Maintaining *unpredictable movement*, whether it's true randomness or adaptive strategies mid-game, is key to preventing opponents from exploiting observable repetitions.

**What assets are provided in the initial codebase?**
The Robocode code-base provides a full environment for developing, running, and visualizing robot battles in Java. The `battles` directory contains scripts and assets related to running matches and managing gameplay logs, while `robots` stores precompiled robot programs that serve as examples or test agents. The `compilers` and `libs` folders include compiled files and necessary libraries for executing and extending the game's functionality. The `config` folder provides configuration files for environment setup, and templates offers starter files to help users design their own robots. Documentation and resources are found in `javadoc`, `ReadMe.html`, and `ReadMe.md`, which describe system components and usage instructions.

**What are the arena configurations?**
In Robocode, the "arena" is called the battlefield and several configuration parameters can be set. For example, the battlefield's default size is 800 × 600 pixels. You can also specify other sizes with the API (width and height between 400 and 5000). The number of rounds that run in a battle can also be specified. The gun cooling rate is the rate at which a robot's gun cools after firing (affects how quickly you can fire again). The inactivity time is how many turns a robot can take without action before being penalised for inactivity. The sentry border size defines how far from the edges sentry robots can move. There is also a flag that determines whether enemy robot names are hidden from the bots. Thus, you can configure the "arena" by choosing size, number of rounds, participants, and rule-modifiers.

**How is the winner determined?** In Robocode battles, the winner is determined primarily by the scoring system. At the end of each round, each robot gets a total score, which includes several components: survival score (bonus for each opponent death while you survive), bullet damage done, ram damage done (if you ram an opponent), last-survivor bonus (if you are the final bot alive). In a multi-round battle, the robot (or team) with the highest cumulative score is considered the winner.

**How are arena logs formatted?**
RoboCode logs summarize the outcome of a set of battles rather than providing turn-by-turn detail.
Each row corresponds to a bot and breaks down its total score into components such as survival points, bonuses, and damage dealt by bullets or ramming.
The logs also record how many times each bot finished in first, second, or third place across the rounds.
Together, this gives a statistical view of performance, highlighting not just who won overall but how they achieved their results.

**Example of RoboCode Logs:**
```
Results for 10 rounds
Robot Name       Total Score   Survival   Surv Bonus   Bullet Dmg   Bullet Bonus 
1st: p2.MyTank*  1362 (55%)    300        60           886          116            0  
2nd: p1.MyTank*  1109 (45%)    200        40           768          101            0        
```

---

### B.7 RobotRumble

RobotRumble is a player-versus-player programming game.
The objective of the competition is quite simple, as summarized on the website (https://robotrumble.org/):

> The rules are simple: (1) two players fight in a match (2) robots spawn every 10 turns (3) a robot can move or attack (4) each robot has 5 health (5) the player with more robots after 100 turns wins

To summarize, RobotRumble is a game that emphasizes the ability to position units effectively and coordinate teams of units to focus on enemy at a time (e.g., if 5 units attack an opposing unit, it takes 1 turn to knock out the unit).

**System Prompt Description of RobotRumble:** You are a software developer ({{player_id}}) competing in a coding game called RobotRumble. RobotRumble is a turn-based coding battle where you program a team of robots in Python to move, attack, and outmaneuver your opponent on a grid. Every decision is driven by your code, and victory comes from crafting logic that positions robots smartly, times attacks well, and adapts over the 100-turn match.

**What are effective strategies?**
First, *avoid getting purged from spawn* by timing your exits — since up to four new robots appear every 10 turns and anything left in spawn is deleted, strong bots step out just before the purge to keep their full roster in play.
Next, take advantage of *movement conflict priority* — when two robots move into the same square, the winner is decided by a fixed clockwise rule, so careful bots choose their approach direction to gain the upper hand.
Finally, practice *focus fire while avoiding friendly fire*: attacks only deal 1 damage but can hit teammates, so good bots coordinate multiple robots to bring down a 5-HP enemy in one turn without accidentally shooting their own.

**How are arena logs formatted?**
RobotRumble logs are displayed as a sequence of ASCII grids (a total of 100 grids per simulation), with numbers marking robot positions and empty cells showing open space.
After each turn, the grid is updated to show new movements, clashes, or unit spawns, giving a clear visual trace of how the battle unfolds.
Below each grid, a summary line shows each player's remaining health and unit counts.

**What assets are provided in the initial codebase?**
The initial codebase includes a command-line interface (CLI) tool (`rumblebot`) that allows users to execute battles between bots directly in the terminal or in a web-based graphical viewer.
The repository also includes example "builtin bots" that can be used as opponents or templates for developing new robots.
Additionally, the repo contains logic scripts and documentation for running matches, viewing results, and managing robot files within the filesystem.

**What are the arena configurations?**
The arena configuration determines the battle environment—typically a rectangular map with fixed dimensions, where robots spawn in random or defined positions.
Each robot operates in discrete turns, executing movement and attack commands according to its programmed logic.
The arena setup remains consistent across matches to ensure fairness.

**How is the winner determined?**
The winner in Robot Rumble is the last surviving team at the end of a match.
Robots can deplete each other's health using attacks while avoiding incoming fire.
If multiple robots remain when the time limit or round limit is reached, the winner is decided based on performance metrics such as remaining health or damage dealt.

**Example of RobotRumble logs:**
```json
{"winner": "Red", "turns": [ {"state": { "objs": {
    "1": {"id": "1", "coords": [0,0], "obj_type": "Terrain", "type": "Wall"},
    "2": {"id": "2", "coords": [0,1], "obj_type": "Terrain", "type": "Wall"},
    "3": {"id": "3", "coords": [0,2], "obj_type": "Terrain", "type": "Wall"},
...
```
