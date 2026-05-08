# Orbit Wars | Kaggle — Tables and Structured Reference Data

**Source:** https://www.kaggle.com/competitions/orbit-wars

---

## Observation Reference

| Field | Type | Description |
|---|---|---|
| planets | [[id, owner, x, y, radius, ships, production], ...] | All planets including comets |
| fleets | [[id, owner, x, y, angle, from_planet_id, ships], ...] | All active fleets |
| player | int | Your player ID (0-3) |
| angular_velocity | float | Planet rotation speed (radians/turn) |
| initial_planets | [[id, owner, x, y, radius, ships, production], ...] | Planet positions at game start |
| comets | [{planet_ids, paths, path_index}, ...] | Active comet group data |
| comet_planet_ids | [int, ...] | Planet IDs that are comets |
| remainingOverageTime | float | Remaining overage time budget (seconds) |

---

## Action Format

Return a list of moves:

```
[[from_planet_id, direction_angle, num_ships], ...]
```

- **from_planet_id**: ID of a planet you own.
- **direction_angle**: Angle in radians (0 = right, pi/2 = down).
- **num_ships**: Integer number of ships to send.

Return an empty list `[]` to take no action.

---

## Configuration

| Parameter | Default | Description |
|---|---|---|
| episodeSteps | 500 | Maximum number of turns |
| actTimeout | 1 | Seconds per turn |
| shipSpeed | 6.0 | Maximum fleet speed |
| sunRadius | 10.0 | Radius of the sun |
| boardSize | 100.0 | Board dimensions |
| cometSpeed | 4.0 | Comet speed (units/turn) |

---

## Planet Data Structure

Each planet is represented as:

```
[id, owner, x, y, radius, ships, production]
```

| Field | Description |
|---|---|
| id | Unique planet identifier |
| owner | Player ID (0-3), or -1 for neutral |
| x | X position in 100x100 continuous space |
| y | Y position in 100x100 continuous space |
| radius | Physical size; determined by production: 1 + ln(production) |
| ships | Current garrison count |
| production | Integer from 1 to 5; ships generated per turn when owned |

---

## Fleet Data Structure

Each fleet is represented as:

```
[id, owner, x, y, angle, from_planet_id, ships]
```

| Field | Description |
|---|---|
| id | Unique fleet identifier |
| owner | Player ID (0-3) |
| x | Current X position |
| y | Current Y position |
| angle | Direction of travel in radians |
| from_planet_id | ID of the planet the fleet was launched from |
| ships | Number of ships in the fleet (does not change during travel) |

---

## Comet Group Data Structure

The `comets` observation field contains comet group data:

```
[{planet_ids, paths, path_index}, ...]
```

| Field | Description |
|---|---|
| planet_ids | Planet IDs belonging to this comet group |
| paths | The full trajectory for each comet |
| path_index | Current position along the path |

---

## Planet Types Summary

| Type | Condition | Behavior |
|---|---|---|
| Orbiting | orbital_radius + planet_radius < 50 | Rotates around sun at 0.025-0.05 radians/turn (randomized per game) |
| Static | Further from center (does not meet orbiting condition) | Does not rotate |

---

## Comet Spawn Schedule

| Spawn Step | Notes |
|---|---|
| 50 | First comet group (4 comets, one per quadrant) |
| 150 | Second comet group |
| 250 | Third comet group |
| 350 | Fourth comet group |
| 450 | Fifth comet group |

All 4 comets in a group share the same starting ship count (random, skewed low — minimum of 4 rolls from 1-99).

---

## Fleet Speed Lookup (Reference Values)

Fleet speed formula: `speed = 1.0 + (maxSpeed - 1.0) * (log(ships) / log(1000)) ^ 1.5`

| Fleet Size | Approximate Speed (units/turn) |
|---|---|
| 1 ship | 1.0 |
| ~500 ships | ~5.0 |
| ~1000 ships | 6.0 (max) |

Maximum speed (shipSpeed) defaults to 6.0 and is configurable.

---

## 4-Fold Symmetry Coordinate Mapping

All planets and comets are placed with 4-fold mirror symmetry around the center:

| Quadrant | Coordinate Transform |
|---|---|
| Q1 (base) | (x, y) |
| Q2 | (100-x, y) |
| Q3 | (x, 100-y) |
| Q4 | (100-x, 100-y) |

This ensures fairness regardless of starting position.

---

## Map Generation Guarantees

| Property | Guarantee |
|---|---|
| Total planets | 20-40 planets |
| Symmetric groups | 5-10 groups of 4 planets |
| Static groups | At least 3 groups guaranteed static |
| Orbiting groups | At least 1 group guaranteed orbiting |
| Starting ships on home planets | 10 ships |
| Starting garrison (non-home) | 5-99 ships (skewed toward lower values) |

---

## Player Starting Positions

| Game Mode | Starting Configuration |
|---|---|
| 2-player (1v1) | Players start on diagonally opposite planets (Q1 and Q4) |
| 4-player (FFA) | Each player gets one planet from the home symmetric group |
