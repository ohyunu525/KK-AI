"""Export static scientific figures from a completed, independently audited study."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".matplotlib-cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from verify_results import audit

BASE = "#35618D"
TUNED = "#CF723D"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, default=HERE / "studies" / "main")
    args = parser.parse_args()
    root = args.study_dir
    verified = audit(root)
    if not verified["final_evaluation_records"]:
        raise RuntimeError("Figures require the completed frozen final evaluation")
    study, selection = read(root / "study.json"), read(root / "selection.json")
    final = read(root / "final" / "result.json")
    winner = selection["selected_candidate_id"]
    fraction = study["spec"]["fractions"][0]
    seeds = selection["seeds"]
    models = study["spec"]["models"]
    records = [read(path) for path in sorted((root / "runs").glob("*/result.json"))]
    index = {(r["configuration"]["tuning"]["candidate_id"], r["configuration"]["model"]["name"],
              r["configuration"]["observation"]["g05_fraction"], r["configuration"]["training"]["seed"]): r
             for r in records}
    plot_dir = root / "figures"
    plot_dir.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "figure.facecolor": "white",
                         "axes.grid": True, "grid.alpha": 0.18, "svg.fonttype": "none"})

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    screening = read(root / "promotion.json")["screening_ranking"]
    ax = axes[0, 0]
    for i, item in enumerate(screening):
        color = TUNED if item["candidate_id"] == winner else BASE if item["candidate_id"] == "baseline" else "#8B939B"
        ax.scatter(item["score"], i, color=color, s=70 if color != "#8B939B" else 35, zorder=3)
        ax.annotate(f"{item['score']:.4f}", (item["score"], i), xytext=(7, 0),
                    textcoords="offset points", va="center", fontsize=9)
    ax.set_yticks(range(len(screening)), [r["candidate_id"] for r in screening])
    ax.invert_yaxis()
    ax.set_xlim(min(r["score"] for r in screening) - 0.005, max(r["score"] for r in screening) + 0.018)
    ax.set_xlabel("Mean best validation structure loss (lower is better)")
    ax.set_title("A   All candidates, screening seed 41", loc="left")

    ax = axes[0, 1]
    ranking = selection["ranking"]
    names = [r["candidate_id"] for r in ranking]
    for seed in seeds:
        y = [r["seed_scores"][str(seed)] for r in ranking]
        ax.plot(range(len(names)), y, marker="o", lw=1, alpha=0.65, label=f"seed {seed}")
    ax.scatter(range(len(names)), [r["score"] for r in ranking], c="black", marker="_", s=250, lw=2.5,
               zorder=5, label="mean")
    ax.set_xticks(range(len(names)), names)
    ax.set_ylabel("Validation structure loss")
    ax.set_title("B   Confirmation: paired seeds, both models", loc="left")
    ax.legend(fontsize=8, frameon=False)

    evaluated = {(r["candidate_id"], r["model"], r["seed"]): r
                 for r in final["records"] if r["split"] == "fresh_test"
                 and r["checkpoint_selection"] == "structure" and r["fraction"] == fraction}
    ax = axes[1, 0]
    for role_index, candidate in enumerate(("baseline", winner)):
        values = [[evaluated[(candidate, model, seed)]["metrics"]["mean_position_3d_error"] for seed in seeds]
                  for model in models]
        x = np.arange(len(models)) + (role_index - 0.5) * 0.34
        means = [float(np.mean(v)) for v in values]
        stds = [float(np.std(v, ddof=1)) for v in values]
        ax.bar(x, means, width=0.3, color=BASE if role_index == 0 else TUNED,
               yerr=stds, capsize=4, label="baseline" if role_index == 0 else winner, alpha=0.86)
        for i, (mean, sd) in enumerate(zip(means, stds)):
            ax.text(x[i], mean + sd + 0.012, f"{mean:.4f}", ha="center", fontsize=9)
    ax.set_xticks(range(len(models)), ["Sign-only", "Full"])
    ax.set_ylim(0, max(ax.get_ylim()[1], 1.1))
    ax.set_ylabel("Mean 3D position error (original coordinate units)")
    ax.set_title(f"C   Fresh holdout position error, G05={fraction:g}", loc="left")
    ax.legend(fontsize=8, frameon=False, loc="lower left")

    ax = axes[1, 1]
    labels, changes, spread = [], [], []
    for model in models:
        for metric, short in (("relative_sign_accuracy", "relative"), ("global_sign_accuracy", "global")):
            if any(evaluated[(candidate, model, seed)]["metrics"][metric] is None
                   for candidate in ("baseline", winner) for seed in seeds):
                continue
            paired = [100 * (evaluated[(winner, model, seed)]["metrics"][metric]
                             - evaluated[("baseline", model, seed)]["metrics"][metric]) for seed in seeds]
            labels.append(("Sign-only" if model == models[0] else "Full") + "\n" + short)
            changes.append(float(np.mean(paired)))
            spread.append(float(np.std(paired, ddof=1)))
    ax.bar(range(len(labels)), changes, yerr=spread, capsize=4,
           color=[TUNED if d >= 0 else "#9B4C64" for d in changes], alpha=0.86)
    ax.axhline(0, color="#3D4349", lw=0.8)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_ylabel("Accuracy change (percentage points; higher is better)")
    ax.set_title("D   Fresh holdout sign tradeoffs", loc="left")
    fig.suptitle(f"ModelExperiment11  |  Selected common setting: {winner}", fontsize=15, fontweight="bold")
    fig.supxlabel("Error bars = sample SD across training seeds, not confidence intervals. "
                  "Test results were computed only after selection was locked.", fontsize=9)
    fig.savefig(plot_dir / "overview.png", dpi=170)
    fig.savefig(plot_dir / "overview.svg")
    plt.close(fig)

    fig, axes = plt.subplots(len(models), len(seeds), figsize=(13, 7), sharey=True, squeeze=False, layout="constrained")
    for row, model in enumerate(models):
        for column, seed in enumerate(seeds):
            ax = axes[row, column]
            for candidate, color in (("baseline", BASE), (winner, TUNED)):
                record = index[(candidate, model, fraction, seed)]
                history = read(root / record["evaluations"]["structure"]["checkpoint_path"].replace("best_structure.pt", "history.json"))
                epochs = [item["epoch"] for item in history]
                ax.plot(epochs, [h["validation"]["structure"] for h in history],
                        color=color, lw=1.5, label=f"{candidate}: validation")
                ax.plot(epochs, [h["train"]["structure"] for h in history],
                        color=color, lw=1, ls="--", alpha=0.6, label=f"{candidate}: train")
                selected = record["evaluations"]["structure"]
                ax.scatter(selected["selected_epoch"], selected["selected_validation_loss"], color=color, s=30, zorder=4)
            ax.set_title(f"{'Sign-only' if row == 0 else 'Full'} / seed {seed}")
            ax.set_xlabel("Epoch")
            if column == 0:
                ax.set_ylabel("Unchanged structure loss")
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle(f"Training and validation curves  |  G05 fraction {fraction:g}", fontsize=15, fontweight="bold")
    fig.supxlabel("Dots show validation-selected epochs; complete histories and both checkpoint selections are retained. "
                  "No test scores are used in these curves.", fontsize=9)
    fig.savefig(plot_dir / "learning_curves.png", dpi=170)
    fig.savefig(plot_dir / "learning_curves.svg")
    plt.close(fig)
    print(plot_dir)


if __name__ == "__main__":
    main()
