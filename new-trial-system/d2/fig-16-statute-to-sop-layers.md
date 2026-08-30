# Figure 16 - One requirement traced through four layers

**Type.** d2-type, layers. **Section.** §5, Legislation.
**Perspective.** *A single requirement, verification before generation, followed
downward through four regulatory layers to the sentence a site operator reads on
the day of surgery, so the legislative work is shown to terminate somewhere
concrete.* No other figure in this paper traces one requirement vertically;
Figure 14 assigns duties across actors at one layer, and Figure 15 traces the
bill's own drafting history rather than its downward effect.

**Caption (2 balanced lines, 76 and 79 characters, numbered as printed).**

```
Figure 16. One requirement through four layers, from a statutory sentence to
the operating-room instruction, with the artifact that carries it at each step.
```

## D2 source

```d2
layers: {
  statute: {
    label: "Layer 1, statute"
    s1: "H. R. 9510, Verification Before Generation in Physical AI Oncology Trials Act" {
      style: { fill: "#800020"; font-color: "#FFFFFF" }
    }
    s2: "Amends the Federal Food, Drug, and Cosmetic Act" { style.fill: "#E2D6D9" }
    s3: "Financial data amendment, cost ledger per verification run" { style.fill: "#E2D6D9" }
  }
  regulation: {
    label: "Layer 2, regulation"
    r1: "21 CFR 312, Physical AI Subpart J" { style: { fill: "#A32A3C"; font-color: "#FFFFFF" } }
    r2: "21 CFR 812 significant risk device pathway" { style.fill: "#E2D6D9" }
    r3: "Adapted ICH E6 R3 clinical practice text" { style.fill: "#E2D6D9" }
  }
  protocol: {
    label: "Layer 3, protocol"
    p1: "Phase 0 simulation validation gate" { style: { fill: "#A32A3C"; font-color: "#FFFFFF" } }
    p2: "At least 1000 simulations across 2 frameworks" { style.fill: "#E2D6D9" }
    p3: "Unified Safety Level at or above 7.0" { style.fill: "#E2D6D9" }
  }
  sop: {
    label: "Layer 4, site standard operating procedure"
    o1: "Pre-incision verification checklist" { style: { fill: "#800020"; font-color: "#FFFFFF" } }
    o2: "Do not generate motion code before the gate clears" { style.fill: "#C9C9C9" }
    o3: "Record the run cost and the verification hash" { style.fill: "#C9C9C9" }
  }
}
```

## TikZ construction table

Absolute coordinates. Canvas 14.8 by 10.4 cm, four stacked bands, drawn top to
bottom because the claim is genuinely a descent.

| Element | Style token | Placement |
|:--|:--|:--|
| Layer 1 band | `d2step`, `fit` its three boxes | y = 0 to -1.85, full width x = 0 to 14.80 |
| Layer 2 band | `d2ghost`, `fit` its three boxes | y = -2.60 to -4.45 |
| Layer 3 band | `d2ghost`, `fit` its three boxes | y = -5.20 to -7.05 |
| Layer 4 band | `d2step`, burgundy stroke 0.9 pt | y = -7.80 to -9.65 |
| Band titles | `d2title` for layers 1 and 4, `d2title2` for layers 2 and 3 | Anchored north west inside each band, 2 mm inset |
| Lead box, each layer | `d2key` for layers 1 and 4, `d2mid` for layers 2 and 3, `text width=42mm` | x = 0.30, vertical center of its band |
| Second box, each layer | `d2soft`, `text width=42mm` | x = 5.15, same y |
| Third box, each layer | `d2soft` for layers 1 to 3, `d2gray2` for layer 4, `text width=42mm` | x = 10.00, same y |
| Descent edges, three | `d2edgeb`, 0.9 pt | From each band's lead box south anchor to the next band's lead box north anchor, at x = 2.40 |
| Descent labels | `d2edge` label, white fill | `carries`, `constrains`, `executes` at the midpoint of each run |
| Layer gutter | 0.75 cm | Between every pair of bands, uniform |
| Provenance strip | `d2mid`, `text width=64mm` | x = 5.15, y = -10.35 |
| In-figure note | `pnote` | x = 0, y = -11.05, `text width=140mm` |

The four bands are the same height, 1.85 cm, and the same width, and the gutter
between them is uniform at 0.75 cm, so the descent reads as four equal steps
rather than as a tapering funnel. Layers 1 and 4 carry the solid burgundy
stroke and layers 2 and 3 the ghost stroke, because the statute and the
operating room are where a requirement is written and where it is obeyed, and
the two layers between are the machinery.

## Layer table

| Layer | Artifact that carries the requirement | Repository source |
|:--|:--|:--|
| 1, statute | H. R. 9510 Bill v5.0, the financial data amendment to the FD&C Act | `new-trial-system/inputs/HR-9510-Bill-v5.zip` |
| 1, statute | VVUQ Physical AI Oncology Trial Bill, statutory text and definitions | `new-trial-system/inputs/VVUQ-Physical-AI-Oncology-Trial-Bill.zip` |
| 2, regulation | 21 CFR 312 Physical AI Subpart J, 21 CFR 812 device pathway, adapted ICH E6(R3) | `national-platform/21cfr312_adapt`, `national-platform/ich_e6r3_adapt` |
| 3, protocol | The Phase 0 gate, with its two quantities | `trial-protocol/final-protocol/publication` |
| 4, site SOP | Pre-incision verification checklist and the two prohibitions | `national-platform/new_trial_psl`, `trial-documents` |

## Edge routing

Three descent edges only, all at a single x of 2.40, all vertical, each 0.75 cm
long, each passing through the uniform band gutter and touching nothing. No
horizontal edges exist, because within a layer the three boxes are read as a
row rather than as a sequence, which is what the layers construct is for. Box
labels are capped at 46 characters and set at `\tiny`, so no label reaches its
box border, and the 0.35 cm horizontal gutter between boxes prevents any two
labels appearing to run together.

## Repository sources

- `new-trial-system/inputs/HR-9510-Bill-v5.zip` - the amendment text, the findings, and the cost ledger requirement that layer 4 records
- `new-trial-system/inputs/VVUQ-Physical-AI-Oncology-Trial-Bill.zip` - statutory text, definitions, prior law, implementation and enforcement
- `new-trial-system/inputs/Earning-the-Clinician's-Trust.zip` - the reliability and oversight questions layer 4 answers at the bedside
- `trial-protocol/final-protocol/publication/LaTeX Source Files.zip` - the Phase 0 gate quantities at layer 3
- `national-platform` - the adapted 21 CFR and ICH text at layer 2
