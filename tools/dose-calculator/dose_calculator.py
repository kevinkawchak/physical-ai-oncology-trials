#!/usr/bin/env python3
"""Dose Calculator: CLI for radiotherapy dose calculations in oncology trials.

Supports Biologically Effective Dose (BED), Equivalent Dose in 2 Gy fractions
(EQD2), Tumor Control Probability (TCP), and Normal Tissue Complication
Probability (NTCP) using the Lyman-Kutcher-Burman model.

Usage:
    python dose_calculator.py bed --dose 60 --fractions 30 --alpha-beta 10
    python dose_calculator.py eqd2 --dose 60 --fractions 30 --alpha-beta 10
    python dose_calculator.py compare --schemes "60/30,42.56/16,34/10" --alpha-beta 10
    python dose_calculator.py tcp --dose 60 --fractions 30 --model poisson --n0 1e9 --alpha 0.3
    python dose_calculator.py ntcp --dose 60 --fractions 30 --td50 50 --m 0.18 --n 0.12

Requirements:
    numpy, scipy (listed in project requirements.txt)

References:
    - Fowler JF. The linear-quadratic formula and progress in fractionated
      radiotherapy. Br J Radiol. 1989;62(740):679-694.
    - Lyman JT. Complication probability as assessed from dose-volume
      histograms. Radiat Res Suppl. 1985;8:S13-19.
    - Kutcher GJ, Burman C. Calculation of complication probability factors
      for non-uniform irradiation. Int J Radiat Oncol Biol Phys. 1989.
    - Webb S, Nahum AE. A model for calculating tumour control probability
      including the effects of inhomogeneous distributions of dose and
      clonogenic cell density. Phys Med Biol. 1993;38(6):653-666.

Note: All illustrative parameters should be validated against your
institution's treatment protocols before clinical use.
"""

import argparse
import json
import math
import sys
from dataclasses import dataclass

try:
    from scipy.special import erf  # noqa: F401
    from scipy.stats import norm

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ---------------------------------------------------------------------------
# Common alpha/beta reference values (Gy) for oncology tissues
# ---------------------------------------------------------------------------
TISSUE_ALPHA_BETA = {
    # Tumors
    "prostate_cancer": 1.5,
    "breast_cancer": 4.0,
    "nsclc": 10.0,
    "head_neck_scc": 10.0,
    "glioblastoma": 10.0,
    "melanoma": 10.0,
    "cervical_cancer": 10.0,
    # Normal tissues (late effects)
    "spinal_cord": 2.0,
    "lung_late": 3.0,
    "brain": 2.0,
    "rectum": 3.0,
    "bladder": 5.0,
    "heart": 2.5,
    "kidney": 2.5,
    "liver": 2.5,
    "optic_nerve": 2.0,
    "brainstem": 2.0,
}

# Common NTCP parameters (LKB model): TD50 (Gy), m, n
# Reference: Emami et al., QUANTEC series
NTCP_PARAMS = {
    "lung_pneumonitis": {"td50": 30.8, "m": 0.37, "n": 0.87},
    "heart_pericarditis": {"td50": 48.0, "m": 0.10, "n": 0.35},
    "liver_rild": {"td50": 39.8, "m": 0.15, "n": 0.97},
    "spinal_cord_myelopathy": {"td50": 66.5, "m": 0.175, "n": 0.05},
    "brainstem_necrosis": {"td50": 65.0, "m": 0.14, "n": 0.16},
    "rectum_grade2": {"td50": 76.9, "m": 0.13, "n": 0.09},
    "parotid_xerostomia": {"td50": 28.4, "m": 0.18, "n": 0.70},
    "kidney_nephritis": {"td50": 28.0, "m": 0.12, "n": 0.70},
    "esophagus_grade3": {"td50": 68.0, "m": 0.11, "n": 0.06},
    "optic_nerve_blindness": {"td50": 65.0, "m": 0.14, "n": 0.25},
}


@dataclass
class DoseResult:
    """Container for dose calculation results."""

    total_dose_gy: float
    fractions: int
    dose_per_fraction_gy: float
    alpha_beta_gy: float
    bed_gy: float = 0.0
    eqd2_gy: float = 0.0
    tcp: float = 0.0
    ntcp: float = 0.0
    model: str = ""

    def to_dict(self) -> dict:
        d = {
            "total_dose_gy": round(self.total_dose_gy, 2),
            "fractions": self.fractions,
            "dose_per_fraction_gy": round(self.dose_per_fraction_gy, 4),
            "alpha_beta_gy": self.alpha_beta_gy,
        }
        if self.bed_gy:
            d["bed_gy"] = round(self.bed_gy, 2)
        if self.eqd2_gy:
            d["eqd2_gy"] = round(self.eqd2_gy, 2)
        if self.tcp:
            d["tcp"] = round(self.tcp, 6)
        if self.ntcp:
            d["ntcp"] = round(self.ntcp, 6)
        if self.model:
            d["model"] = self.model
        return d


def calc_bed(total_dose: float, fractions: int, alpha_beta: float) -> float:
    """Calculate Biologically Effective Dose (BED).

    BED = nd(1 + d/(alpha/beta))
    where n = number of fractions, d = dose per fraction.
    """
    d = total_dose / fractions
    return total_dose * (1 + d / alpha_beta)


def calc_eqd2(total_dose: float, fractions: int, alpha_beta: float) -> float:
    """Calculate Equivalent Dose in 2 Gy fractions (EQD2).

    EQD2 = BED / (1 + 2/(alpha/beta))
    """
    bed = calc_bed(total_dose, fractions, alpha_beta)
    return bed / (1 + 2.0 / alpha_beta)


def calc_tcp_poisson(
    total_dose: float,
    fractions: int,
    n0: float,
    alpha: float,
    alpha_beta: float,
) -> float:
    """Calculate TCP using the Poisson model.

    TCP = exp(-N0 * exp(-alpha * BED))
    where BED accounts for fractionation via the LQ model.
    """
    bed = calc_bed(total_dose, fractions, alpha_beta)
    surviving = n0 * math.exp(-alpha * bed)
    return math.exp(-surviving)


def calc_tcp_logistic(
    total_dose: float,
    fractions: int,
    td50: float,
    gamma50: float,
    alpha_beta: float,
) -> float:
    """Calculate TCP using the logistic (sigmoid) model.

    TCP = 1 / (1 + (TD50/EQD2)^(4*gamma50))
    """
    eqd2 = calc_eqd2(total_dose, fractions, alpha_beta)
    if eqd2 <= 0:
        return 0.0
    exponent = 4.0 * gamma50
    return 1.0 / (1.0 + (td50 / eqd2) ** exponent)


def calc_ntcp_lkb(
    total_dose: float,
    fractions: int,
    td50: float,
    m: float,
    n: float,
    alpha_beta: float = 3.0,
    volume_fraction: float = 1.0,
) -> float:
    """Calculate NTCP using the Lyman-Kutcher-Burman (LKB) model.

    For uniform irradiation of a partial volume:
        EUD = D * v^(-1/n)   (generalized equivalent uniform dose)
        t = (EUD - TD50) / (m * TD50)
        NTCP = Phi(t)        (standard normal CDF)
    """
    eqd2 = calc_eqd2(total_dose, fractions, alpha_beta)

    # Generalized EUD for partial volume irradiation
    if n > 0 and volume_fraction < 1.0:
        eud = eqd2 * (volume_fraction ** (-1.0 / n)) if volume_fraction > 0 else eqd2
    else:
        eud = eqd2

    t = (eud - td50) / (m * td50) if (m * td50) != 0 else 0.0

    if HAS_SCIPY:
        return float(norm.cdf(t))
    else:
        # Fallback: approximate normal CDF using error function
        return 0.5 * (1 + math.erf(t / math.sqrt(2)))


def _parse_scheme(scheme_str: str) -> tuple[float, int]:
    """Parse a fractionation scheme string like '60/30' into (dose, fractions)."""
    parts = scheme_str.strip().split("/")
    if len(parts) != 2:
        raise ValueError(f"Invalid scheme format '{scheme_str}'. Use 'dose/fractions' (e.g., '60/30')")
    return float(parts[0]), int(parts[1])


# ---------------------------------------------------------------------------
# CLI command handlers
# ---------------------------------------------------------------------------


def cmd_bed(args):
    """Calculate BED for a fractionation scheme."""
    bed = calc_bed(args.dose, args.fractions, args.alpha_beta)
    d_per_fx = args.dose / args.fractions
    eqd2 = calc_eqd2(args.dose, args.fractions, args.alpha_beta)

    result = DoseResult(
        total_dose_gy=args.dose,
        fractions=args.fractions,
        dose_per_fraction_gy=d_per_fx,
        alpha_beta_gy=args.alpha_beta,
        bed_gy=bed,
        eqd2_gy=eqd2,
    )

    print("=" * 50)
    print("BIOLOGICALLY EFFECTIVE DOSE (BED)")
    print("=" * 50)
    print(f"  Prescription:    {args.dose:.2f} Gy in {args.fractions} fractions")
    print(f"  Dose/fraction:   {d_per_fx:.4f} Gy")
    print(f"  alpha/beta:      {args.alpha_beta:.1f} Gy")
    print(f"  BED:             {bed:.2f} Gy")
    print(f"  EQD2:            {eqd2:.2f} Gy")

    if args.output:
        _write_json(args.output, result.to_dict())
        print(f"\nReport written to {args.output}")


def cmd_eqd2(args):
    """Calculate EQD2 for a fractionation scheme."""
    eqd2 = calc_eqd2(args.dose, args.fractions, args.alpha_beta)
    bed = calc_bed(args.dose, args.fractions, args.alpha_beta)
    d_per_fx = args.dose / args.fractions

    print("=" * 50)
    print("EQUIVALENT DOSE IN 2 Gy FRACTIONS (EQD2)")
    print("=" * 50)
    print(f"  Prescription:    {args.dose:.2f} Gy in {args.fractions} fractions")
    print(f"  Dose/fraction:   {d_per_fx:.4f} Gy")
    print(f"  alpha/beta:      {args.alpha_beta:.1f} Gy")
    print(f"  BED:             {bed:.2f} Gy")
    print(f"  EQD2:            {eqd2:.2f} Gy")

    if args.output:
        result = DoseResult(
            total_dose_gy=args.dose,
            fractions=args.fractions,
            dose_per_fraction_gy=d_per_fx,
            alpha_beta_gy=args.alpha_beta,
            bed_gy=bed,
            eqd2_gy=eqd2,
        )
        _write_json(args.output, result.to_dict())
        print(f"\nReport written to {args.output}")


def cmd_compare(args):
    """Compare multiple fractionation schemes."""
    schemes_str = args.schemes.split(",")
    alpha_beta = args.alpha_beta

    results = []
    for s in schemes_str:
        dose, fx = _parse_scheme(s)
        bed = calc_bed(dose, fx, alpha_beta)
        eqd2 = calc_eqd2(dose, fx, alpha_beta)
        results.append({
            "scheme": s.strip(),
            "dose_gy": dose,
            "fractions": fx,
            "dose_per_fx_gy": round(dose / fx, 4),
            "bed_gy": round(bed, 2),
            "eqd2_gy": round(eqd2, 2),
        })

    print("=" * 70)
    print(f"FRACTIONATION COMPARISON (alpha/beta = {alpha_beta} Gy)")
    print("=" * 70)
    print(f"  {'Scheme':<12} {'Dose(Gy)':<10} {'Fx':<5} {'d/fx(Gy)':<10} {'BED(Gy)':<10} {'EQD2(Gy)':<10}")
    print("-" * 70)
    for r in results:
        print(
            f"  {r['scheme']:<12} {r['dose_gy']:<10.2f} {r['fractions']:<5} "
            f"{r['dose_per_fx_gy']:<10.4f} {r['bed_gy']:<10.2f} {r['eqd2_gy']:<10.2f}"
        )

    # Also compute for late-effect tissue (alpha/beta = 3)
    if alpha_beta != 3.0:
        print()
        print("  Late-effect comparison (alpha/beta = 3.0 Gy):")
        print(f"  {'Scheme':<12} {'BED_late(Gy)':<14} {'EQD2_late(Gy)':<14}")
        print("-" * 50)
        for r in results:
            bed_late = calc_bed(r["dose_gy"], r["fractions"], 3.0)
            eqd2_late = calc_eqd2(r["dose_gy"], r["fractions"], 3.0)
            print(f"  {r['scheme']:<12} {bed_late:<14.2f} {eqd2_late:<14.2f}")

    if args.output:
        report = {
            "comparison_type": "fractionation",
            "alpha_beta_gy": alpha_beta,
            "schemes": results,
        }
        _write_json(args.output, report)
        print(f"\nReport written to {args.output}")


def cmd_tcp(args):
    """Calculate Tumor Control Probability."""
    d_per_fx = args.dose / args.fractions

    if args.model == "poisson":
        if args.n0 is None or args.alpha is None:
            print("ERROR: Poisson TCP requires --n0 and --alpha", file=sys.stderr)
            sys.exit(1)
        tcp = calc_tcp_poisson(args.dose, args.fractions, args.n0, args.alpha, args.alpha_beta)
        model_detail = f"Poisson (N0={args.n0:.2e}, alpha={args.alpha})"
    elif args.model == "logistic":
        if args.td50 is None or args.gamma50 is None:
            print("ERROR: Logistic TCP requires --td50 and --gamma50", file=sys.stderr)
            sys.exit(1)
        tcp = calc_tcp_logistic(args.dose, args.fractions, args.td50, args.gamma50, args.alpha_beta)
        model_detail = f"Logistic (TD50={args.td50}, gamma50={args.gamma50})"
    else:
        print(f"ERROR: Unknown TCP model '{args.model}'", file=sys.stderr)
        sys.exit(1)

    bed = calc_bed(args.dose, args.fractions, args.alpha_beta)
    eqd2 = calc_eqd2(args.dose, args.fractions, args.alpha_beta)

    print("=" * 50)
    print("TUMOR CONTROL PROBABILITY (TCP)")
    print("=" * 50)
    print(f"  Prescription:    {args.dose:.2f} Gy in {args.fractions} fractions")
    print(f"  Dose/fraction:   {d_per_fx:.4f} Gy")
    print(f"  alpha/beta:      {args.alpha_beta:.1f} Gy")
    print(f"  BED:             {bed:.2f} Gy")
    print(f"  EQD2:            {eqd2:.2f} Gy")
    print(f"  Model:           {model_detail}")
    print(f"  TCP:             {tcp:.4f} ({tcp*100:.2f}%)")

    if args.output:
        result = DoseResult(
            total_dose_gy=args.dose,
            fractions=args.fractions,
            dose_per_fraction_gy=d_per_fx,
            alpha_beta_gy=args.alpha_beta,
            bed_gy=bed,
            eqd2_gy=eqd2,
            tcp=tcp,
            model=model_detail,
        )
        _write_json(args.output, result.to_dict())
        print(f"\nReport written to {args.output}")


def cmd_ntcp(args):
    """Calculate Normal Tissue Complication Probability."""
    d_per_fx = args.dose / args.fractions

    # Use preset if specified
    if args.organ:
        if args.organ not in NTCP_PARAMS:
            print(f"ERROR: Unknown organ preset '{args.organ}'", file=sys.stderr)
            print(f"Available presets: {', '.join(sorted(NTCP_PARAMS.keys()))}", file=sys.stderr)
            sys.exit(1)
        params = NTCP_PARAMS[args.organ]
        td50 = params["td50"]
        m = params["m"]
        n = params["n"]
        organ_label = args.organ
    else:
        if args.td50 is None or args.m is None or args.n is None:
            print("ERROR: NTCP requires --td50, --m, --n (or --organ preset)", file=sys.stderr)
            sys.exit(1)
        td50 = args.td50
        m = args.m
        n = args.n
        organ_label = "custom"

    alpha_beta_late = args.alpha_beta if args.alpha_beta else 3.0
    volume_fraction = args.volume if args.volume else 1.0

    ntcp = calc_ntcp_lkb(args.dose, args.fractions, td50, m, n, alpha_beta_late, volume_fraction)
    bed = calc_bed(args.dose, args.fractions, alpha_beta_late)
    eqd2 = calc_eqd2(args.dose, args.fractions, alpha_beta_late)

    print("=" * 50)
    print("NORMAL TISSUE COMPLICATION PROBABILITY (NTCP)")
    print("=" * 50)
    print(f"  Prescription:    {args.dose:.2f} Gy in {args.fractions} fractions")
    print(f"  Dose/fraction:   {d_per_fx:.4f} Gy")
    print(f"  alpha/beta:      {alpha_beta_late:.1f} Gy (late effects)")
    print(f"  Organ/Endpoint:  {organ_label}")
    print(f"  TD50:            {td50:.1f} Gy")
    print(f"  m:               {m}")
    print(f"  n:               {n}")
    print(f"  Volume fraction: {volume_fraction:.2f}")
    print(f"  BED (late):      {bed:.2f} Gy")
    print(f"  EQD2 (late):     {eqd2:.2f} Gy")
    print(f"  NTCP:            {ntcp:.6f} ({ntcp*100:.3f}%)")

    if args.output:
        result = {
            "total_dose_gy": args.dose,
            "fractions": args.fractions,
            "dose_per_fraction_gy": round(d_per_fx, 4),
            "alpha_beta_gy": alpha_beta_late,
            "organ": organ_label,
            "td50_gy": td50,
            "m": m,
            "n": n,
            "volume_fraction": volume_fraction,
            "bed_gy": round(bed, 2),
            "eqd2_gy": round(eqd2, 2),
            "ntcp": round(ntcp, 6),
        }
        _write_json(args.output, result)
        print(f"\nReport written to {args.output}")


def cmd_tissues(args):
    """List available tissue alpha/beta values and NTCP presets."""
    print("=" * 60)
    print("TISSUE alpha/beta VALUES (Gy)")
    print("=" * 60)
    print(f"  {'Tissue':<25} {'alpha/beta (Gy)':<15}")
    print("-" * 45)
    for tissue, ab in sorted(TISSUE_ALPHA_BETA.items()):
        print(f"  {tissue:<25} {ab:<15.1f}")

    print()
    print("=" * 60)
    print("NTCP PRESETS (LKB Model Parameters)")
    print("=" * 60)
    print(f"  {'Organ/Endpoint':<30} {'TD50(Gy)':<10} {'m':<8} {'n':<8}")
    print("-" * 60)
    for organ, params in sorted(NTCP_PARAMS.items()):
        print(f"  {organ:<30} {params['td50']:<10.1f} {params['m']:<8.3f} {params['n']:<8.3f}")


def _write_json(filepath: str, data: dict):
    """Write data to a JSON file."""
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        prog="dose_calculator",
        description="Radiotherapy Dose Calculator: BED, EQD2, TCP, NTCP calculations for oncology trial protocol design.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # bed
    p_bed = subparsers.add_parser("bed", help="Calculate Biologically Effective Dose")
    p_bed.add_argument("--dose", type=float, required=True, help="Total dose in Gy")
    p_bed.add_argument("--fractions", type=int, required=True, help="Number of fractions")
    p_bed.add_argument("--alpha-beta", type=float, required=True, help="alpha/beta ratio in Gy")
    p_bed.add_argument("--output", "-o", help="Write JSON report to file")

    # eqd2
    p_eqd2 = subparsers.add_parser("eqd2", help="Calculate Equivalent Dose in 2 Gy fractions")
    p_eqd2.add_argument("--dose", type=float, required=True, help="Total dose in Gy")
    p_eqd2.add_argument("--fractions", type=int, required=True, help="Number of fractions")
    p_eqd2.add_argument("--alpha-beta", type=float, required=True, help="alpha/beta ratio in Gy")
    p_eqd2.add_argument("--output", "-o", help="Write JSON report to file")

    # compare
    p_compare = subparsers.add_parser("compare", help="Compare fractionation schemes")
    p_compare.add_argument("--schemes", required=True, help="Comma-separated schemes (e.g., '60/30,42.56/16,34/10')")
    p_compare.add_argument("--alpha-beta", type=float, required=True, help="alpha/beta ratio in Gy")
    p_compare.add_argument("--output", "-o", help="Write JSON report to file")

    # tcp
    p_tcp = subparsers.add_parser("tcp", help="Calculate Tumor Control Probability")
    p_tcp.add_argument("--dose", type=float, required=True, help="Total dose in Gy")
    p_tcp.add_argument("--fractions", type=int, required=True, help="Number of fractions")
    p_tcp.add_argument("--alpha-beta", type=float, required=True, help="alpha/beta ratio in Gy")
    p_tcp.add_argument("--model", choices=["poisson", "logistic"], default="poisson", help="TCP model (default: poisson)")
    p_tcp.add_argument("--n0", type=float, help="Initial clonogen count (Poisson model)")
    p_tcp.add_argument("--alpha", type=float, help="Radiosensitivity alpha (Poisson model)")
    p_tcp.add_argument("--td50", type=float, help="Dose for 50%% TCP (logistic model)")
    p_tcp.add_argument("--gamma50", type=float, help="Slope at 50%% TCP (logistic model)")
    p_tcp.add_argument("--output", "-o", help="Write JSON report to file")

    # ntcp
    p_ntcp = subparsers.add_parser("ntcp", help="Calculate Normal Tissue Complication Probability (LKB model)")
    p_ntcp.add_argument("--dose", type=float, required=True, help="Total dose in Gy")
    p_ntcp.add_argument("--fractions", type=int, required=True, help="Number of fractions")
    p_ntcp.add_argument("--organ", help="Use preset LKB parameters for organ (see 'tissues' command)")
    p_ntcp.add_argument("--td50", type=float, help="TD50 in Gy (or use --organ preset)")
    p_ntcp.add_argument("--m", type=float, help="Slope parameter m (or use --organ preset)")
    p_ntcp.add_argument("--n", type=float, help="Volume parameter n (or use --organ preset)")
    p_ntcp.add_argument("--alpha-beta", type=float, default=3.0, help="alpha/beta for late effects (default: 3.0 Gy)")
    p_ntcp.add_argument("--volume", type=float, default=1.0, help="Irradiated volume fraction 0-1 (default: 1.0)")
    p_ntcp.add_argument("--output", "-o", help="Write JSON report to file")

    # tissues
    subparsers.add_parser("tissues", help="List available tissue alpha/beta values and NTCP presets")

    args = parser.parse_args()

    commands = {
        "bed": cmd_bed,
        "eqd2": cmd_eqd2,
        "compare": cmd_compare,
        "tcp": cmd_tcp,
        "ntcp": cmd_ntcp,
        "tissues": cmd_tissues,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
