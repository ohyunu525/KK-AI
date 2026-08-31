"""검증 전용 pilot, 기존 36개 학습 이력, 소스 보존 여부를 재현 가능하게 집계한다."""

from __future__ import annotations

import ast
import json
import os
import statistics
import sys
import tempfile
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / "Codes"))
import ModelExperiment10 as experiment

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "kk-ai-m10-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class WithoutDocstrings(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
            node.body.pop(0)
        return node

    visit_ClassDef = visit_FunctionDef


def source_api(filename: str) -> dict[str, str]:
    tree = ast.parse((ROOT / "Codes" / filename).read_text(encoding="utf-8"))
    tree = WithoutDocstrings().visit(tree)
    functions = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            functions[node.name] = ast.dump(node)
        elif isinstance(node, ast.ClassDef):
            for method in node.body:
                if isinstance(method, ast.FunctionDef):
                    functions[f"{node.name}.{method.name}"] = ast.dump(method)
    return functions


def main() -> None:
    histories, summaries = {}, []
    for directory in ("screening_seed43", "confirmation_seeds41_42", "confirmation_sign_only43"):
        rows = json.loads((OUT / directory / "summary.json").read_text(encoding="utf-8"))
        for row in rows:
            key = row["model"], row["seed"], row["variant"]
            if key in histories:
                raise ValueError(f"Duplicate pilot: {key}")
            histories[key] = json.loads((OUT / directory / f"{row['run_name']}.json").read_text(encoding="utf-8"))
            summaries.append(row)
    expected = {(model, seed, variant) for model in experiment.DEFAULT_MODELS
                for seed in (41, 42, 43) for variant in ("baseline", "dropout010")}
    if expected.difference(histories):
        raise RuntimeError(f"Confirmation is not finished: {sorted(expected.difference(histories))}")
    if any(h["fraction"] != 1.0 for h in histories.values()):
        raise ValueError("This summary compares the fixed G05=100% pilot only")
    pairs = []
    for model in experiment.DEFAULT_MODELS:
        for seed in (41, 42, 43):
            baseline, dropout = (histories[model, seed, variant] for variant in ("baseline", "dropout010"))
            pairs.append({
                "model": model, "seed": seed, "fraction": 1.0,
                **{f"{label}_{key}": result[key] for label, result in (("baseline", baseline), ("dropout010", dropout))
                   for key in ("best_structure_loss", "best_structure_epoch", "best_total_loss", "best_total_epoch", "epochs_completed")},
                **{f"{objective}_improvement_pct": 100 * (baseline[f"best_{objective}_loss"] - dropout[f"best_{objective}_loss"])
                   / baseline[f"best_{objective}_loss"] for objective in experiment.CHECKPOINT_SELECTIONS},
            })
    means = {}
    for model in experiment.DEFAULT_MODELS:
        rows = [r for r in pairs if r["model"] == model]
        means[model] = {
            "paired_seed_count": len(rows),
            **{f"{label}_mean_best_{objective}_loss": statistics.mean(r[f"{label}_best_{objective}_loss"] for r in rows)
               for label in ("baseline", "dropout010") for objective in experiment.CHECKPOINT_SELECTIONS},
            **{f"mean_paired_{objective}_improvement_pct": statistics.mean(r[f"{objective}_improvement_pct"] for r in rows)
               for objective in experiment.CHECKPOINT_SELECTIONS},
            "structure_improved_seed_count": sum(r["structure_improvement_pct"] > 0 for r in rows),
            "structure_improvement_sample_std_pct": statistics.stdev(r["structure_improvement_pct"] for r in rows),
        }
    # 새 제어 클래스를 실제 과거 이력에 적용한다. 물리/데이터 재학습은 하지 않는다.
    replay_rows = []
    for seed in (41, 42, 43):
        directory = ROOT / "Results" / "new_learning9_experiments" / f"5point_routing_v1_seed{seed}" / "runs"
        for path in sorted(directory.glob("*/history.json")):
            history = json.loads(path.read_text(encoding="utf-8"))
            if len(history) != 300:
                raise ValueError(f"Unexpected historical run: {path}")
            tracker = experiment.DualObjectiveEarlyStopping(experiment.RegularizationSettings(early_stopping_patience=20))
            for row in history:
                if tracker.update(row["epoch"], row["validation"]):
                    break
            original_bests = {objective: min(history, key=lambda h: h["validation"][objective])["epoch"]
                              for objective in experiment.CHECKPOINT_SELECTIONS}
            replay_rows.append({"run_id": path.parent.name, "seed": seed, "stop_epoch": tracker.epoch,
                                "original_epochs": len(history), **{f"best_{k}_epoch": v for k, v in original_bests.items()},
                                "both_best_checkpoints_preserved": max(original_bests.values()) <= tracker.epoch})
    if len(replay_rows) != 36 or not all(r["both_best_checkpoints_preserved"] for r in replay_rows):
        raise AssertionError("Historical early stopping preservation check failed")
    old_api, new_api = source_api("ModelExperiment9.py"), source_api("ModelExperiment10.py")
    preservation = {
        "old_function_count": len(old_api), "new_function_count": len(new_api),
        "removed": sorted(set(old_api).difference(new_api)), "added": sorted(set(new_api).difference(old_api)),
        "unchanged": sorted(name for name in old_api.keys() & new_api.keys() if old_api[name] == new_api[name]),
        "adapted": sorted(name for name in old_api.keys() & new_api.keys() if old_api[name] != new_api[name]),
        "run_epoch_is_physics_alias": experiment.run_epoch is experiment.physics.run_epoch,
        "evaluate_model_is_physics_alias": experiment.evaluate_model is experiment.physics.evaluate_model,
        "source_sha256": {name: experiment.file_sha256(ROOT / "Codes" / name)
                          for name in ("ModelExperiment9.py", "ModelExperiment10.py", "NewLearning9.py")},
        "note": "API/AST checks supplement, but do not replace, behavioral regression tests.",
    }
    assert not preservation["removed"]
    historical = {
        "runs": len(replay_rows), "both_best_preserved": sum(r["both_best_checkpoints_preserved"] for r in replay_rows),
        "original_epochs": sum(r["original_epochs"] for r in replay_rows),
        "early_stopping_epochs": sum(r["stop_epoch"] for r in replay_rows),
        "stop_epoch_min": min(r["stop_epoch"] for r in replay_rows),
        "stop_epoch_max": max(r["stop_epoch"] for r in replay_rows),
    }
    historical["epoch_reduction_pct"] = 100 * (1 - historical["early_stopping_epochs"] / historical["original_epochs"])
    result = {
        "scope": "train 8000, validation 1000, G05 fraction 1.0; test set unused; pilot max 100 epochs, patience 20",
        "pilot_runs": len(histories), "confirmation_pairs": pairs, "model_means": means,
        "historical_replay": historical,
        "baseline43_matches_original_structure_minimum_exactly": histories["g05_full_reconstruction", 43, "baseline"]["best_structure_loss"]
            == 0.5975169048309327,
        "limitations": ["One fixed data split", "Three training seeds, not independent data samples",
                        "No test-set claim", "Other G05 fractions have functional tests, not a full performance sweep",
                        "Reported training loss has dropout enabled when p>0, unlike validation loss"],
    }
    experiment.atomic_write_json(OUT / "validation_summary.json", result)
    experiment.atomic_write_json(OUT / "preservation_audit.json", preservation)
    experiment.atomic_write_csv(OUT / "paired_validation.csv", pairs)
    experiment.atomic_write_csv(OUT / "historical_replay_v10.csv", replay_rows)
    experiment.atomic_write_csv(OUT / "pilot_runs.csv", [{k: v for k, v in r.items() if k != "parameter_count"} for r in summaries])
    make_plot(histories, pairs)
    print(json.dumps({"pilot_runs": len(histories), "model_means": means, "historical_replay": historical,
                      "removed_functions": preservation["removed"]}, indent=2))


def make_plot(histories, pairs) -> None:
    plt.rcParams.update({"font.family": "Malgun Gothic", "font.size": 10, "axes.unicode_minus": False,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "#f7f9fc", "axes.facecolor": "white", "savefig.facecolor": "#f7f9fc"})
    blue, teal, gray, orange = "#365ca8", "#138779", "#8b96a5", "#c8762d"
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.8))
    fig.subplots_adjust(top=.83, bottom=.14, hspace=.55, wspace=.28)
    fig.suptitle("ModelExperiment10 · 과적합 제어 검증", x=.06, y=.965, ha="left", fontsize=21, weight="bold")
    fig.text(.06, .916, "학습 8,000 / 검증 1,000 · G05 32개 · 동일 분할과 학습 seed 41·42·43 · 테스트셋 미사용", color="#546372", fontsize=11)
    ax = axes[0, 0]
    for variant, label, color in (("baseline", "기존 설정", blue), ("dropout010", "Dropout 10%", teal)):
        result = histories["g05_full_reconstruction", 43, variant]
        epoch = [r["epoch"] for r in result["history"]]
        for phase, style, suffix in (("train", "--", "학습"), ("validation", "-", "검증")):
            ax.plot(epoch, [r[phase]["structure"] for r in result["history"]], color=color, ls=style, lw=1.6,
                    label=f"{label} · {suffix}")
        ax.scatter(result["best_structure_epoch"], result["best_structure_loss"], color=color, s=30, zorder=5)
    ax.set(title="Full / seed 43: 구조 손실과 최적 epoch", xlabel="Epoch", ylabel="구조 손실", ylim=(.35, .95))
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax = axes[0, 1]
    for variant, label, color in (("baseline", "기존", blue), ("dropout010", "Dropout 10%", teal),
                                 ("dropout020", "Dropout 20%", orange), ("decay001", "Weight decay 0.001", gray)):
        result = histories["g05_full_reconstruction", 43, variant]
        ax.plot([r["epoch"] for r in result["history"]], [r["validation"]["structure"] for r in result["history"]],
                color=color, lw=1.4, label=f"{label} · 최저 {result['best_structure_loss']:.4f}")
    ax.set(title="첫 탐색: 검증 손실만으로 후보 비교", xlabel="Epoch", ylabel="검증 구조 손실", ylim=(.55, .75))
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1, 0]
    for index, model in enumerate(experiment.DEFAULT_MODELS):
        for row in [r for r in pairs if r["model"] == model]:
            ax.plot([index * 2, index * 2 + 1], [row["baseline_best_structure_loss"], row["dropout010_best_structure_loss"]],
                    marker="o", lw=1.4, color={41: blue, 42: orange, 43: teal}[row["seed"]],
                    label=f"seed {row['seed']}" if index == 0 else None)
    ax.set(xticks=[0, 1, 2, 3], xticklabels=["Sign-only\n기존", "Sign-only\n10%", "Full\n기존", "Full\n10%"],
           title="각 seed의 최저 검증 구조 손실", ylabel="낮을수록 좋음")
    ax.legend(frameon=False, fontsize=8, ncol=3)
    ax = axes[1, 1]
    for index, row in enumerate(pairs):
        ax.bar(index - .17, row["baseline_epochs_completed"], .32, color=blue, label="기존 + 조기 종료" if index == 0 else None)
        ax.bar(index + .17, row["dropout010_epochs_completed"], .32, color=teal, label="10% + 조기 종료" if index == 0 else None)
    ax.set(xticks=range(6), xticklabels=[f"{'Sign' if r['model']=='g05_sign_only' else 'Full'}\n{r['seed']}" for r in pairs],
           title="실제 실행한 epoch (최대 100, patience 20)", ylabel="Epoch", ylim=(0, 100))
    ax.legend(frameon=False, fontsize=8)
    for ax in axes.flat:
        ax.grid(axis="y", color="#e3e8ee", lw=.6)
        ax.set_axisbelow(True)
    fig.text(.06, .057, "드롭아웃 학습 손실은 마스크를 적용한 값이라 검증 손실과 직접 같은 조건이 아닙니다.\n"
             "한 데이터 분할의 검증 결과입니다. 별도 테스트 성능 향상이나 모든 G05 비율의 개선을 보장하지 않습니다.",
             ha="left", color="#546372", fontsize=9)
    for extension in ("png", "svg"):
        fig.savefig(OUT / f"regularization_validation.{extension}", dpi=170, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
