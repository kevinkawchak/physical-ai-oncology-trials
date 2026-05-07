# Chunk 07 — Tables

---

## Table 1: Main Results — Elo Ratings Per Model Per Arena

| Model | BattleSnake | CoreWar | Halite | Poker | RoboCode | RobotRumble | **Overall** |
|---|---|---|---|---|---|---|---|
| Claude Sonnet 4.5 | 1470 | 1641 | 1408 | 1248 | 1361 | 1423 | **1389** |
| GPT-5 | 1339 | 1199 | 1522 | 1599 | 1409 | 1293 | **1360** |
| o3 | 1357 | 1348 | 1576 | 1277 | 1338 | 1309 | **1343** |
| Claude Sonnet 4 | 1253 | 1339 | 1111 | 1233 | 1033 | 1361 | **1223** |
| GPT-5 Mini | 1369 | 926 | 1185 | 1429 | 1217 | 1092 | **1200** |
| Gemini 2.5 Pro | 1115 | 1043 | 1186 | 978 | 1315 | 1044 | **1125** |
| Grok Code Fast | 833 | 1170 | 824 | 886 | 1033 | 1016 | **1004** |
| Qwen3 Coder | 860 | 929 | 784 | 945 | 890 | 1057 | **952** |

*Caption:* Elo ratings per model per arena.

*Label:* tab:main_results

---

## Table 2: Code Arenas Currently Implemented in CodeClash

| Arena | Description | n | Language |
|---|---|---|---|
| Battlesnake | Grid-based survival and territory control | 2+ | Python |
| Core War | Assembly programs competing in shared memory | 2+ | Redcode |
| Halite | Resource collection and territory expansion on grid | 2+ | Multiple |
| Poker | No-limit Texas Hold'em | 2+ | Python |
| RoboCode | Tank duels with movement, scanning, and firing | 2+ | Java |
| RobotRumble | Turn-based grid battles with spawning robots | 2 | JavaScript |

*Caption:* Code arenas currently implemented in CodeClash. Arenas represent a diverse landscape of objectives (e.g., eliminate opponents, accumulate money/resources), programming languages, and challenges (e.g., decipher opponent strategy from logs, decide how to adapt code, manage growing codebase). n is number of players.

*Label:* tab:list_arenas

---

## Table 3: TrueSkill Ratings — 6-Player Core War

| Model | μ |
|---|---|
| Claude Sonnet 4.5 | 28.38 ± 0.65 |
| o3 | 27.11 ± 0.64 |
| Grok Code Fast | 25.65 ± 0.65 |
| GPT-5 | 24.76 ± 0.64 |
| Gemini 2.5 Pro | 23.62 ± 0.65 |
| Qwen3 Coder | 22.30 ± 0.66 |

*Caption:* TrueSkill ratings per model based on 20 tournaments of 6-player Core War. TrueSkill models each player's skill as a Gaussian distribution with mean μ (skill estimate) and standard deviation σ (uncertainty). After each round, both parameters are updated based on match outcomes: winning increases μ while exceeding expectations, and σ decreases as the system gains confidence in the estimate. Final placement (1st, 2nd, ..., 6th) determines rating updates.

*Label:* tab:trueskill

---

## Table 4: Rank Stability Metrics (Bootstrapping Experiments)

| Metric | Nonparametric | Parametric |
|---|---|---|
| Kendall's τ | 0.966 | 0.956 |
| Spearman's ρ | 0.988 | 0.984 |
| Footrule (normalized) | 0.030 | 0.038 |
| Top-1 consistency | 0.896 | 0.839 |
| Pairwise order agreement | 0.983 | 0.978 |

*Caption:* Rank stability metrics of the Elo-based ranking of LMs over all arenas based on bootstrapping experiments.

*Label:* tab:rank_stability_bootstrapping

---

## Table 5: Elo Ratings With Uncertainties

| Model | BattleSnake | CoreWar | Halite | Poker | RoboCode | RobotRumble | All |
|---|---|---|---|---|---|---|---|
| Claude Sonnet 4.5 | 1470 ± 52 | 1641 ± 73 | 1408 ± 50 | 1248 ± 44 | 1361 ± 43 | 1423 ± 47 | **1389 ± 18** |
| GPT-5 | 1339 ± 44 | 1199 ± 43 | 1522 ± 56 | 1599 ± 64 | 1409 ± 46 | 1293 ± 41 | **1360 ± 17** |
| o3 | 1357 ± 45 | 1348 ± 47 | 1576 ± 60 | 1277 ± 46 | 1338 ± 43 | 1309 ± 42 | **1343 ± 17** |
| Claude Sonnet 4 | 1253 ± 45 | 1339 ± 46 | 1111 ± 48 | 1233 ± 44 | 1033 ± 45 | 1361 ± 43 | **1223 ± 16** |
| GPT-5 Mini | 1369 ± 45 | 926 ± 50 | 1185 ± 47 | 1429 ± 50 | 1217 ± 41 | 1092 ± 41 | **1200 ± 16** |
| Gemini 2.5 Pro | 1115 ± 45 | 1043 ± 45 | 1186 ± 47 | 978 ± 48 | 1315 ± 42 | 1044 ± 44 | **1125 ± 16** |
| Grok Code Fast | 833 ± 63 | 1170 ± 43 | 824 ± 63 | 886 ± 54 | 1033 ± 45 | 1016 ± 46 | **1004 ± 18** |
| Qwen3 Coder | 860 ± 59 | 929 ± 51 | 784 ± 67 | 945 ± 53 | 890 ± 55 | 1057 ± 43 | **952 ± 20** |

*Caption:* ELO ratings with uncertainties.

*Label:* tab:elo_ratings_uncertainties
