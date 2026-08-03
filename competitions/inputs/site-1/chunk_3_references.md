# Orbit Wars | Kaggle — AGENTS.md, Code Examples, and Citation

**Source:** https://www.kaggle.com/competitions/orbit-wars

---

## AGENTS.md

### Orbit Wars: Getting Started

This guide walks you through building an agent, testing it locally, and submitting it to the Orbit Wars competition on Kaggle.

#### Game Overview

Orbit Wars is a real-time strategy game on a 100x100 board with a sun at the center. Players conquer planets by sending fleets of ships between them.

- **Planets** produce ships each turn (proportional to their radius)
- **Inner planets** rotate around the central sun; outer planets are static
- **Fleets** fly in straight lines at a given angle from their source planet
- **Fleet speed** scales with fleet size (1 ship = 1/turn, larger fleets up to 6/turn)
- **Combat**: arriving fleet ships are subtracted from the planet's garrison. If the garrison drops below 0, ownership flips
- **Sun**: fleets that hit the sun are destroyed
- **Comets**: temporary planets that fly through the board on elliptical paths
- **Win condition**: highest ship count (planets + fleets) when time runs out, or last player standing

See How to Play Orbit Wars for full rules and configuration defaults.

#### Your Agent

Your agent is a function that receives an observation and returns a list of moves.

Observation fields:
- `player` — your player ID (0-3)
- `planets` — list of `[id, owner, x, y, radius, ships, production]` (owner -1 = neutral)
- `fleets` — list of `[id, owner, x, y, angle, from_planet_id, ships]`
- `angular_velocity` — rotation speed of inner planets (radians/turn)

Action format: Each move is `[from_planet_id, angle_in_radians, num_ships]`.

#### Example — Nearest Planet Sniper

```python
import math
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet


def agent(obs):
    moves = []
    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    raw_planets = obs.get("planets", []) if isinstance(obs, dict) else obs.planets
    planets = [Planet(*p) for p in raw_planets]

    my_planets = [p for p in planets if p.owner == player]
    targets = [p for p in planets if p.owner != player]
    if not targets:
        return moves
    for mine in my_planets:
        # Find nearest planet we don't own
        nearest = min(targets, key=lambda t: math.hypot(mine.x - t.x, mine.y - t.y))
        # Send exactly enough ships to capture it
        ships_needed = nearest.ships + 1
        if mine.ships >= ships_needed:
            angle = math.atan2(nearest.y - mine.y, nearest.x - mine.x)
            moves.append([mine.id, angle, ships_needed])
    return moves
```

#### Agent Convenience

The module exports named tuples for easier field access:

```python
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet, CENTER, ROTATION_RADIUS_LIMIT


def agent(obs):
    planets = [Planet(*p) for p in obs.get("planets", [])]
    fleets = [Fleet(*f) for f in obs.get("fleets", [])]
    player = obs.get("player", 0)
    for p in planets:
        print(p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production)

    return []  # list of [from_planet_id, angle, num_ships]
```

---

## Test Locally

Install the environment from PyPI (Orbit Wars requires version 1.28.0 or later):

```bash
pip install "kaggle-environments>=1.28.0"
```

Run a game from Python or a notebook:

```python
from kaggle_environments import make

env = make("orbit_wars", configuration={"seed": 42}, debug=True)
env.run(["main.py", "random"])
# View result
final = env.steps[-1]
for i, s in enumerate(final):
    print(f"Player {i}: reward={s.reward}, status={s.status}")
# Render in a notebook
env.render(mode="ipython", width=800, height=600)
```

---

## Set Up the Kaggle CLI

Install the CLI:

```bash
pip install kaggle
```

You'll need a Kaggle account — sign up at https://www.kaggle.com if you don't have one. Then download your API credentials at https://www.kaggle.com/settings/api by clicking "Generate New Token" under the "API" section.

Recommended: API token file. Save the token string to `~/.kaggle/access_token`:

```bash
mkdir -p ~/.kaggle
# Paste the token from the Kaggle settings UI into this file
nano ~/.kaggle/access_token
chmod 600 ~/.kaggle/access_token
```

Alternative auth methods:

- OAuth (browser flow): `kaggle auth login`
- Environment variable: `export KAGGLE_API_TOKEN=xxxxxxxxxxxxxx`

Verify the CLI is wired up:

```bash
kaggle competitions list -s "orbit wars"
```

---

## Find the Competition

```bash
kaggle competitions list -s "orbit wars"
kaggle competitions pages orbit-wars
kaggle competitions pages orbit-wars --content
```

---

## Accept the Competition Rules

Before submitting, you **must** accept the rules on the Kaggle website. Navigate to https://www.kaggle.com/competitions/orbit-wars and click "Join Competition".

Verify you've joined:

```bash
kaggle competitions list --group entered
```

---

## Download Competition Data

```bash
kaggle competitions download orbit-wars -p orbit-wars-data
```

---

## Submit Your Agent

Your submission must have a `main.py` at the root with an `agent` function.

Single file agent:

```bash
kaggle competitions submit orbit-wars -f main.py -m "Nearest planet sniper v1"
```

Multi-file agent — bundle into a tar.gz with `main.py` at the root:

```bash
tar -czf submission.tar.gz main.py helper.py model_weights.pkl
kaggle competitions submit orbit-wars -f submission.tar.gz -m "Multi-file agent v1"
```

Notebook submission:

```bash
kaggle competitions submit orbit-wars -k YOUR_USERNAME/orbit-wars-agent -f submission.tar.gz -v 1 -m "Notebook agent v1"
```

---

## Monitor Your Submission

Check submission status:

```bash
kaggle competitions submissions orbit-wars
```

Note the submission ID from the output — you'll need it for episodes.

---

## List Episodes

Once your submission has played some games:

```bash
kaggle competitions episodes <SUBMISSION_ID>
```

CSV output for scripting:

```bash
kaggle competitions episodes <SUBMISSION_ID> -v
```

---

## Download Replays and Logs

Download the replay JSON for an episode (for visualization or analysis):

```bash
kaggle competitions replay <EPISODE_ID>
kaggle competitions replay <EPISODE_ID> -p ./replays
```

Download agent logs to debug your agent's behavior:

```bash
# Logs for the first agent (index 0)
kaggle competitions logs <EPISODE_ID> 0
# Logs for the second agent (index 1)
kaggle competitions logs <EPISODE_ID> 1 -p ./logs
```

---

## Check the Leaderboard

```bash
kaggle competitions leaderboard orbit-wars -s
```

---

## Typical Workflow

```bash
# Test locally
python -c "
from kaggle_environments import make
env = make('orbit_wars', debug=True)
env.run(['main.py', 'random'])
print([(i, s.reward) for i, s in enumerate(env.steps[-1])])
"
# Submit
kaggle competitions submit orbit-wars -f main.py -m "v1"
# Check status
kaggle competitions submissions orbit-wars

# Review episodes
kaggle competitions episodes <SUBMISSION_ID>
# Download replay and logs
kaggle competitions replay <EPISODE_ID>
kaggle competitions logs <EPISODE_ID> 0
# Check leaderboard
kaggle competitions leaderboard orbit-wars -s
```

---

## Citation

Bovard Doerschuk-Tiberi, Walter Reade, and Addison Howard. Orbit Wars. https://kaggle.com/competitions/orbit-wars, 2026. Kaggle.
