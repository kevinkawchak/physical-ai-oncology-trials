# Figure 10 - The NSF Funding Ladder as a Layered Stack

**Platform.** D2. **Native construct.** A layered stack of containers, each
reached only through the one beneath it.

## Perspective no other figure in this day gives

Table 16 lists the tiers and pilots side by side, which makes them look like
alternatives. They are not. Each is reached only by winning the one below it,
and the stack makes that visible: the company stands on the lowest layer, and the
pilot the inquiry concerns sits two gates above the first award it could win.

## Native source

```d2
direction: up
pitch: "Project Pitch | No funding; four fields; two a year" { style.fill: "#4B2A7B"; style.font-color: white }
phase1: "SBIR Phase I | Up to $305K, 6 to 18 months" { style.fill: "#EAE4F3" }
phase2: "SBIR Phase II | Up to $1.25M, about 24 months" { style.fill: "#EAE4F3" }
phase2b: "Phase IIB supplement | $50K to $500K, with a match" { style.fill: "#F2F3F4" }
ncti: "Commercialization Readiness Pilot | $1M to $3.75M each; 6 to 8 firms" { style.fill: "#F2F3F4" }
breakthrough: "Strategic Breakthrough | Up to $30M, 1 to 1 match" { style.fill: "#F2F3F4" }
pitch -> phase1: "gate: invitation after the Pitch" { style.stroke-dash: 3 }
phase1 -> phase2: "gate: a Phase I award" { style.stroke-dash: 3 }
phase2 -> phase2b: "gate: a Phase II award" { style.stroke-dash: 3 }
phase2b -> ncti: "gate: Phase II or IIB" { style.stroke-dash: 3 }
ncti -> breakthrough: "gate: Phase II and a match" { style.stroke-dash: 3 }
here: "The company is here" { style.fill: "#4B2A7B"; style.font-color: white }
here -> pitch
```

The pilot and the Strategic Breakthrough tier are both open to Phase II
awardees, so their order in the stack is by size, not by a gate between them;
the dashed gate labels say which award opens each layer.

## TikZ construction

| Element | Style | Geometry |
|:--|:--|:--|
| Six layers | A rounded frame 10.9 cm wide per layer, with two `d2key`, `d2soft` or `d2gray` boxes inside it | Rows 1.05 cm apart from y = 0.00 to 5.25 |
| Tier name | Left box, 48 mm, bold | x = 0.00 |
| Amount | Right box, 48 mm | x = 5.30 |
| Gates | Dashed `d2edged` ticks with a gray label | x = 11.15, at each boundary between two layers |
| Company marker | `d2key`, 24 mm, with a `d2edgeb` arrow to the lowest layer | `(13.35,0.00)` |
| Legend | `\legkey` three times | y = -0.95 |

## Caption, exactly as set

> Figure 10. The NSF tiers and pilots as a layered stack, each opened only by an award
> or a match on the layer below, with the company on the Pitch layer at the bottom.

The two lines are 84 and 81 characters.

## Where it is used

`../packet/sections/sec-01-the-inquiry-and-the-gate.tex`, after Table 16.
