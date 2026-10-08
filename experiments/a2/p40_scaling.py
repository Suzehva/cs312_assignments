"""Exact exponent bookkeeping for the handout's Problem 4.0.

Derivation only: no model construction, training, network calls, or TeX build.
This checks the stated alignment assumptions; it does not establish that
real optimizer updates satisfy those assumptions.
"""

from fractions import Fraction
import json
from pathlib import Path


HALF = Fraction(1, 2)
ALIGNMENTS = {"full": Fraction(1), "none": HALF}
OUT_PATH = Path(__file__).resolve().parent / "results" / "p40_scaling.json"


def triplet(a, b, c):
    return {"a": str(a), "b": str(b), "c": str(c)}


def derive(update_alignment, readout_alignment):
    alpha = ALIGNMENTS[update_alignment]
    omega = ALIGNMENTS[readout_alignment]
    hidden = (Fraction(0), HALF, alpha)
    readout = (alpha, omega - alpha, Fraction(0))
    inputs = (Fraction(0), Fraction(0), Fraction(0))
    h_a, h_b, h_c = hidden
    r_a, r_b, r_c = readout
    i_a, i_b, i_c = inputs
    powers = {
        "input_initial_feature": -i_a - i_b,
        "input_direct_update": -i_a - i_c,
        "hidden_initial_feature": HALF - h_a - h_b,
        "hidden_direct_update": alpha - h_a - h_c,
        "hidden_initial_weight_feature_change_bound": HALF - h_a - h_b,
        "hidden_interaction_bound": alpha - h_a - h_c,
        "readout_initial_logits": HALF - r_a - r_b,
        "readout_direct_update": alpha - r_a - r_c,
        "readout_initial_weight_feature_change": omega - r_a - r_b,
        "readout_interaction_bound": alpha - r_a - r_c,
    }
    if any(power != 0 for key, power in powers.items() if key != "readout_initial_logits"):
        raise ValueError("An order-one requirement or interaction bound failed.")
    if powers["readout_initial_logits"] > 0:
        raise ValueError("Initial logits grow with width.")
    # All prescriptions must reproduce the handout's baseline at m=1.
    # Fixed aspect ratio: each hidden fan-in k grows proportionally to width.
    reference_width, reference_fan_in, base_lr = 512, 512, .003
    for m in (1, 2, 4, 16):
        hidden_variance = (1 / reference_fan_in) * m**float(-2*h_b)
        if hidden_variance != 1 / (reference_fan_in*m):
            raise ValueError("Hidden initialization is not 1/k.")
        readout_multiplier = m**float(-r_a)
        readout_variance = (1 / reference_width) * m**float(-2*r_b)
        # Variance of each initial logit for an RMS-one independent input:
        logit_variance = readout_multiplier**2 * reference_width*m * readout_variance
        expected_variance = m**float(1 - 2*omega)
        if abs(logit_variance - expected_variance) > 1e-12:
            raise ValueError("Forward and stored initialization scales disagree.")
        if m == 1:
            if (readout_multiplier, readout_variance, base_lr*m**float(-h_c)) != (
                    1., 1/reference_width, base_lr):
                raise ValueError("A rule fails to match the reference model.")
    return {
        "update_alignment": update_alignment,
        "readout_feature_change_alignment": readout_alignment,
        "alpha": str(alpha), "omega_move": str(omega),
        "abc": {"hidden": triplet(*hidden), "readout": triplet(*readout),
                "input": triplet(*inputs)},
        "relative_width_powers": {
            "input": {"forward": "0", "initialization_variance": "0", "lr": "0"},
            "hidden": {"forward": str(-h_a), "initialization_variance": str(-2*h_b),
                       "lr": str(-h_c)},
            "readout": {"forward": str(-r_a), "initialization_variance": str(-2*r_b),
                        "lr": str(-r_c)},
        },
        "width_powers_of_terms": {k: str(v) for k, v in powers.items()},
        "recovers_handout_mup": alpha == omega == 1,
    }


def main():
    cases = [derive(u, r) for u in ALIGNMENTS for r in ALIGNMENTS]
    report = {
        "source": "worksheets/hparam_invariants/units/mup_no_alignment.tex",
        "assumptions": [
            "Fixed step, base LR and reference width; normalized Adam direction RMS is Theta(1).",
            "Pre-update inputs and feature changes have Theta(1) RMS.",
            "R(U, x0)=Theta(n**alpha); S(V0, delta_x)=Theta(n**omega_move).",
            "Interaction additionally assumes R(U, delta_x)=O(n**alpha).",
            "Hidden dimensions have a fixed aspect ratio; readout class count and input fan-in stay fixed.",
            "Hidden and input a=0; readout c=0, as required by the handout.",
        ],
        "reference_initialization_variances": {"input": "1/d", "hidden": "1/k0",
                                               "readout": "1/n0"},
        "cases": cases,
        "checks": "Exact rational exponents; bounded logits and interactions; finite-width reference matching.",
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    for row in cases:
        print(f"Update={row['update_alignment']}, readout={row['readout_feature_change_alignment']}: "
              f"hidden={row['abc']['hidden']}, readout={row['abc']['readout']}")
    print(f"Saved {OUT_PATH}; no training launched.")


if __name__ == "__main__":
    main()
    # uv run python -m experiments.a2.p40_scaling
