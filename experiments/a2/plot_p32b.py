"""Offline source-WD fits and available P3.2(b) target results."""

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from experiments.a2.plot_p1a import PLOT_DIR, set_style
from experiments.a2.p32b_results import RESULTS_DIR, SOURCE_PATH, TARGET_PATH, PREDICTION_PATH
from experiments.a2.p32b_lr_predictions import PREDICTED_LRS

import matplotlib.pyplot as plt
import numpy as np


ANALYSIS_PATH = RESULTS_DIR / "p32b_analysis.json"
OUT_PATH = PLOT_DIR / "p32b_wd_source_fits.png"
WRITEUP_DIR = Path(__file__).resolve().parents[2] / "6abdc5b0bad58b38dfd83f81"


def analyze_sources(rows):
    batches = []
    for batch in (8, 16, 32, 64):
        matching = sorted((r for r in rows if r["batch_size"] == batch),
                          key=lambda r: r["weight_decay"])
        if len(matching) != 7:
            raise ValueError(f"Expected seven WD settings at B={batch}.")
        wds = np.array([r["weight_decay"] for r in matching])
        losses = np.array([r["final_val_loss"] for r in matching])
        x = np.log2(wds / .1)
        coefficients = np.polyfit(x, losses, 2)
        a, b, _ = coefficients
        wd_opt = float(.1 * 2**(-b / (2*a)))
        if a <= 0 or not wds.min() < wd_opt < wds.max():
            raise ValueError(f"No interior quadratic minimum at B={batch}.")
        best_index = int(losses.argmin())
        if best_index in (0, len(matching) - 1):
            raise ValueError(f"Measured minimum is at a sweep boundary at B={batch}.")
        local = np.polyfit(x[best_index-1:best_index+2], losses[best_index-1:best_index+2], 2)
        local_opt = float(.1 * 2**(-local[1] / (2*local[0])))
        batches.append({
            "batch_size": batch, "quadratic_coefficients": coefficients.tolist(),
            "fitted_optimal_wd": wd_opt,
            "fitted_minimum_loss": float(np.polyval(coefficients, np.log2(wd_opt / .1))),
            "fit_rmse": float(np.sqrt(np.mean((losses - np.polyval(coefficients, x))**2))),
            "local_three_point_optimal_wd": local_opt,
            "best_measured_wd": float(wds[best_index]),
            "best_measured_loss": float(losses[best_index]), "runs": matching,
        })
    x = np.log(np.array([r["batch_size"] for r in batches]) / 64)
    y = np.log([r["fitted_optimal_wd"] for r in batches])
    exponent, intercept = np.polyfit(x, y, 1)
    law = {"reference_batch": 64, "wd_ref": float(np.exp(intercept)),
           "exponent": float(exponent), "doubling_multiplier": float(2**exponent),
           "r_squared_log": float(1 - np.sum((y-intercept-exponent*x)**2) /
                                  np.sum((y-y.mean())**2))}
    local_y = np.log([r["local_three_point_optimal_wd"] for r in batches])
    local_exponent, local_intercept = np.polyfit(x, local_y, 1)
    local_law = {"reference_batch": 64, "wd_ref": float(np.exp(local_intercept)),
                 "exponent": float(local_exponent),
                 "predictions": [
                     {"batch_size": b,
                      "weight_decay": float(np.exp(local_intercept) * (b/64)**local_exponent)}
                     for b in (128, 256)
                 ]}
    return {
        "wd_loss_fit": "L_B = a_B*y^2 + b_B*y + c_B; y=log2(WD/0.1); all seven WDs",
        "source_batches": batches, "wd_rule": law,
        "local_three_point_rule_sensitivity": local_law,
        "wd_predictions": [
            {"batch_size": b, "learning_rate": .0015,
             "weight_decay": law["wd_ref"] * (b/64)**law["exponent"]}
            for b in (128, 256)
        ],
        "lr_predictions": [
            {"batch_size": b, "learning_rate": lr, "weight_decay": .1}
            for b, lr in PREDICTED_LRS.items()
        ],
    }


def freeze_predictions(report):
    prediction = {"wd_rule": report["wd_rule"],
                  "wd_predictions": report["wd_predictions"],
                  "lr_predictions": report["lr_predictions"]}
    if PREDICTION_PATH.exists():
        saved = json.loads(PREDICTION_PATH.read_text())
        if any(saved[key] != value for key, value in prediction.items()):
            raise ValueError("Source fits changed; do not overwrite frozen target predictions.")
        return
    prediction["recorded_at_utc"] = datetime.now(timezone.utc).isoformat()
    prediction["recording_note"] = "WD source law recorded before fetching target losses."
    PREDICTION_PATH.write_text(json.dumps(prediction, indent=2) + "\n")


def plot(report):
    set_style()
    fig, (loss_ax, wd_ax) = plt.subplots(1, 2, figsize=(12.6, 4.7))
    colors = plt.get_cmap("viridis")(np.linspace(.08, 1.0, 4))
    for row, color in zip(report["source_batches"], colors):
        wds = np.array([r["weight_decay"] for r in row["runs"]])
        losses = np.array([r["final_val_loss"] for r in row["runs"]])
        dense = np.geomspace(wds.min(), wds.max(), 250)
        loss_ax.scatter(wds, losses, color=color, edgecolors="#333333", linewidths=.4,
                        s=35, zorder=3)
        loss_ax.plot(dense, np.polyval(row["quadratic_coefficients"], np.log2(dense / .1)),
                     color=color, label=f"B = {row['batch_size']}")
        loss_ax.scatter(row["fitted_optimal_wd"], row["fitted_minimum_loss"], marker="*",
                        color=color, s=140, edgecolors="#333333", linewidths=.5, zorder=4)
        wd_ax.scatter(row["batch_size"], row["fitted_optimal_wd"], marker="*", color=color,
                      s=140, edgecolors="#333333", linewidths=.5, zorder=4)
    loss_ax.set_xscale("log", base=2)
    loss_ax.set_xticks([.025, .1, .4, 1.6], labels=[".025", ".1", ".4", "1.6"])
    loss_ax.set_xlabel("Weight decay (peak LR = .0015)")
    loss_ax.set_ylabel("Final validation loss")
    loss_ax.set_title("Source WD sweeps")
    loss_ax.legend(frameon=False)
    law = report["wd_rule"]
    dense = np.geomspace(8, 256, 250)
    wd_ax.plot(dense, law["wd_ref"] * (dense/64)**law["exponent"], linestyle="--",
               color=colors[2], linewidth=1.7)
    for pred in report["wd_predictions"]:
        wd_ax.scatter(pred["batch_size"], pred["weight_decay"], marker="D", s=65,
                      facecolors="none", edgecolors=plt.get_cmap("viridis")(1.0),
                      linewidths=1.8, zorder=3)
        wd_ax.annotate(f"{pred['weight_decay']:.3f}",
                       (pred["batch_size"], pred["weight_decay"]),
                       xytext=(-7, 8), textcoords="offset points", ha="right", fontsize=10)
    wd_ax.set_xscale("log", base=2)
    wd_ax.set_yscale("log")
    wd_ax.set_xticks([8, 16, 32, 64, 128, 256], labels=["8", "16", "32", "64", "128", "256"])
    wd_ax.set_xlabel("Batch size (sequences/update)")
    wd_ax.set_ylabel("Fitted optimal / predicted WD")
    wd_ax.set_title("Source law and untested WD predictions")
    wd_ax.text(.03, .97, rf"$\lambda^*={law['wd_ref']:.4f}(B/64)^{{{law['exponent']:.3f}}}$",
               transform=wd_ax.transAxes, va="top", fontsize=11)
    for ax in (loss_ax, wd_ax):
        ax.grid(True, linestyle=":", alpha=.3)
    fig.tight_layout(w_pad=2)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH)
    plt.close(fig)
    destination = WRITEUP_DIR / "figures" / OUT_PATH.name
    shutil.copy2(OUT_PATH, destination)
    print(f"Saved {OUT_PATH} and copied to {destination}.")


def main():
    snapshot = json.loads(SOURCE_PATH.read_text())
    report = analyze_sources(snapshot["runs"])
    freeze_predictions(report)
    if TARGET_PATH.exists():
        report["lr_target_runs"] = json.loads(TARGET_PATH.read_text())["runs"]
    report["wd_target_status"] = "not launched or measured; comparison incomplete"
    ANALYSIS_PATH.write_text(json.dumps(report, indent=2) + "\n")
    plot(report)
    for row in report["source_batches"]:
        print(f"B={row['batch_size']}: WD*={row['fitted_optimal_wd']:.9f}, "
              f"local sensitivity={row['local_three_point_optimal_wd']:.9f}, "
              f"fit RMSE={row['fit_rmse']:.6f}")
    print(json.dumps(report["wd_rule"], indent=2))


if __name__ == "__main__":
    main()
    # uv run python -m experiments.a2.plot_p32b
