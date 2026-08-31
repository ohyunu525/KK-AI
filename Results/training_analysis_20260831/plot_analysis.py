"""Generate static research figures from the audit outputs, without inference."""

from pathlib import Path
import csv
import json
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "kk-ai-training-audit-matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
plt.rcParams.update({"font.family": "Malgun Gothic", "font.size": 10,
                     "axes.unicode_minus": False, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titleweight": "bold",
                     "axes.titlesize": 12, "figure.facecolor": "#f7f9fc",
                     "axes.facecolor": "white", "savefig.facecolor": "#f7f9fc",
                     "grid.color": "#dfe4eb", "grid.linewidth": .6})
BLUE, RED, TEAL, GRAY = "#2563a6", "#cd5a48", "#18867d", "#76849a"
summary = json.loads((OUT / "audit_summary.json").read_text(encoding="utf-8"))
diagnostics = json.loads((OUT / "prediction_diagnostics.json").read_text(encoding="utf-8"))
with (OUT / "saved_metrics.csv").open(encoding="utf-8-sig") as handle:
    saved = list(csv.DictReader(handle))
with (OUT / "paired_comparisons.csv").open(encoding="utf-8-sig") as handle:
    paired = list(csv.DictReader(handle))
with (OUT / "error_strata.csv").open(encoding="utf-8-sig") as handle:
    strata = list(csv.DictReader(handle))
with (OUT / "calibration.csv").open(encoding="utf-8-sig") as handle:
    calibration = list(csv.DictReader(handle))
run = summary["last_completed_run"]["run_id"]
history = json.loads((ROOT / "Results/new_learning9_experiments/5point_routing_v1_seed43/runs" / run / "history.json").read_text())
best = summary["last_completed_run"]["training_result"]["best_structure_epoch"]
epoch = np.arange(1, 301)


def save(fig, name, footer):
    fig.text(.035, .019, footer, ha="left", va="bottom", color="#566579", fontsize=9)
    fig.savefig(OUT / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


fig, axes = plt.subplots(2, 2, figsize=(13, 8.2))
fig.subplots_adjust(top=.86, bottom=.12, hspace=.47, wspace=.29)
fig.suptitle("300 epoch까지 학습했지만, 최적 모델은 26 epoch", x=.055, y=.965, ha="left", fontsize=20, weight="bold")
fig.text(.055, .915, "마지막 완료 실행 · seed 43 · G05 32개 · full reconstruction · 시험 표본 1,000개", color="#52637a", fontsize=11)
ax = axes[0, 0]
for phase, label, color in (("train", "훈련", BLUE), ("validation", "검증", RED)):
    ax.plot(epoch, [h[phase]["structure"] for h in history], label=label, color=color, lw=1.6)
ax.axvline(best, color=TEAL, ls="--", lw=1)
ax.scatter([best, 300], [history[best - 1]["validation"]["structure"], history[-1]["validation"]["structure"]], color=RED, zorder=3)
ax.annotate("선택: 0.598", (best, .598), (65, .42), arrowprops={"arrowstyle": "-", "color": GRAY}, color=TEAL)
ax.annotate("마지막: 1.003", (300, 1.003), (175, 1.16), arrowprops={"arrowstyle": "-", "color": GRAY}, color=RED)
ax.set(xlabel="Epoch", ylabel="구조 손실", title="훈련 오차와 검증 오차의 분리", xlim=(1, 300), ylim=(0, 1.3))
ax.legend(frameon=False, loc="upper left", ncol=2)
ax.grid(axis="y")
ax = axes[0, 1]
for key, label, color in (("position", "위치", BLUE), ("magnitude", "전하 크기", TEAL), ("relative_sign", "상대부호", RED)):
    ax.plot(epoch, [h["validation"][key] for h in history], label=label, color=color, lw=1.5)
ax.axvline(best, color=GRAY, ls="--", lw=1)
ax.set(xlabel="Epoch", ylabel="검증 손실 성분", title="위치·상대부호 손실이 함께 증가", xlim=(1, 300), ylim=(0, .85))
ax.legend(frameon=False, ncol=3, loc="upper right")
ax.grid(axis="y")
ax = axes[1, 0]
for phase, label, color in (("train", "훈련", BLUE), ("validation", "검증", RED)):
    ax.plot(epoch, [h[phase]["global_sign"] for h in history], color=color, label=label, lw=1.6)
ax.axvline(best, color=GRAY, ls="--", lw=1)
ax.set(xlabel="Epoch", ylabel="전체부호 BCE", title="전체부호는 비교적 안정적", xlim=(1, 300), ylim=(.2, .72))
ax.legend(frameon=False)
ax.grid(axis="y")
ax = axes[1, 1]
keys = ["mean_position_mae", "mean_position_3d_error", "charge_magnitude_mae", "charge_mae"]
names = ["위치 MAE", "3D 거리", "크기 MAE", "전하 MAE"]
growth = [(diagnostics["full_latest"][k]["mean"] / diagnostics["full_best"][k]["mean"] - 1) * 100 for k in keys]
bars = ax.barh(names[::-1], growth[::-1], color=RED, height=.56)
ax.bar_label(bars, labels=[f"+{v:.1f}%" for v in growth[::-1]], padding=5)
ax.set(xlabel="26 epoch 대비 300 epoch 시험 오차 증가 (%)", title="체크포인트 재평가로 확인한 성능 저하", xlim=(0, 28))
ax.grid(axis="x")
ax.set_axisbelow(True)
save(fig, "training_dynamics", "선택은 검증 손실로 수행. 마지막 epoch 시험 평가는 이번 사후 진단에서만 추가했으며 기존 선택을 변경하지 않았음.")


fig, axes = plt.subplots(2, 2, figsize=(13, 8.2))
fig.subplots_adjust(top=.86, bottom=.13, hspace=.48, wspace=.30)
fig.suptitle("G05를 구조 예측에 연결한 효과는 작고 일관되지 않음", x=.055, y=.965, ha="left", fontsize=19, weight="bold")
fig.text(.055, .915, "구조 손실로 선택한 체크포인트 · 양수는 full 개선 · seed별로 환경이 달라 통합 평균을 내지 않음", color="#52637a", fontsize=10.5)
for ax, metric, title in ((axes[0, 0], "mean_position_mae", "위치 MAE 개선율: seed에 따라 방향이 다름"),
                          (axes[0, 1], "charge_magnitude_mae", "전하 크기 MAE: seed 42·43에서 소폭 개선")):
    for seed, color in ((41, RED), (42, BLUE), (43, TEAL)):
        rows = sorted([r for r in paired if r["seed"] == str(seed) and r["selection"] == "structure" and r["metric"] == metric], key=lambda r: float(r["fraction"]))
        ax.plot([int(r["sensor_count"]) for r in rows], [float(r["relative_improvement_percent"]) for r in rows], "o-", color=color, label=f"seed {seed}", lw=1.6, ms=4)
    ax.axhline(0, color=GRAY, lw=1)
    ax.set(xlabel="관측 G05 센서 수", ylabel="sign-only 대비 개선 (%)", title=title, xticks=[0, 3, 8, 16, 24, 32])
    ax.legend(frameon=False, ncol=3, fontsize=9)
    ax.grid(axis="y")
ax = axes[1, 0]
for model, label, color in (("g05_sign_only", "sign-only", BLUE), ("g05_full_reconstruction", "full", TEAL)):
    rows = sorted([r for r in saved if r["seed"] == "43" and r["selection"] == "structure" and r["model"] == model and float(r["fraction"]) > 0], key=lambda r: float(r["fraction"]))
    ax.plot([int(r["sensor_count"]) for r in rows], [float(r["global_sign_accuracy"]) * 100 for r in rows], "o-", color=color, label=label)
ax.set(xlabel="관측 G05 센서 수", ylabel="전체부호 정확도 (%)", title="전체부호: 센서 증가 효과는 빠르게 둔화", xticks=[3, 8, 16, 24, 32], ylim=(82, 90))
ax.legend(frameon=False)
ax.grid(axis="y")
ax = axes[1, 1]
with (OUT / "paired_bootstrap_seed43.csv").open(encoding="utf-8-sig") as handle:
    boot = {r["metric"]: r for r in csv.DictReader(handle) if float(r["fraction"]) == 1}
for y, metric, label, color in ((1, "mean_position_mae", "위치 MAE", BLUE), (0, "charge_magnitude_mae", "크기 MAE", TEAL)):
    r = boot[metric]
    mean, lo, hi = (float(r[k]) for k in ("improvement_mean", "ci95_low", "ci95_high"))
    ax.errorbar(mean, y, xerr=[[mean - lo], [hi - mean]], fmt="o", color=color, capsize=5, lw=2)
ax.axvline(0, color=GRAY, ls="--")
ax.set(xlabel="sign-only 오차 - full 오차 (각 데이터 단위)", yticks=[0, 1], yticklabels=["크기 MAE", "위치 MAE"], ylim=(-.6, 1.6), title="seed 43·32개: 위치 개선 구간은 0을 포함")
ax.grid(axis="x")
save(fig, "routing_comparison", "오차막대: 시험 샘플 단위 paired bootstrap 95% 구간, 10,000회. 고정된 두 모델에 조건부이며 seed 간 재현성을 뜻하지 않음.")


fig, axes = plt.subplots(2, 2, figsize=(13, 8.2))
fig.subplots_adjust(top=.86, bottom=.13, hspace=.47, wspace=.30)
fig.suptitle("높은 부호 정확도만으로 정밀한 복원을 판단할 수 없음", x=.055, y=.965, ha="left", fontsize=19, weight="bold")
fig.text(.055, .915, "seed 43 · G05 32개 · 기존 지표의 정의를 유지하고 별도 진단을 추가", color="#52637a", fontsize=11)
ax = axes[0, 0]
labels = ["전하별 상대부호", "5개 상대부호 모두"]
x = np.arange(2)
joints = [diagnostics["full_best"][k]["mean"] * 100 for k in ("relative_sign_accuracy", "relative_configuration_accuracy")]
geos = [diagnostics["full_best"][k]["mean"] * 100 for k in ("geometric_relative_sign_accuracy", "geometric_relative_configuration_accuracy")]
for xx, values, label, color in ((x - .18, joints, "공동 매칭 (기존)", BLUE), (x + .18, geos, "위치만으로 매칭", TEAL)):
    bars = ax.bar(xx, values, .34, color=color, label=label)
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
ax.set(xticks=x, xticklabels=labels, ylabel="정확도 (%)", ylim=(0, 111), title="대응 결정에 부호를 쓰는지에 따라 달라짐")
ax.legend(frameon=False, ncol=2, fontsize=9, loc="lower center")
ax.grid(axis="y")
ax.set_axisbelow(True)
ax = axes[0, 1]
for label, title, color in (("sign_only_best", "sign-only · 31 epoch", GRAY), ("full_best", "full · 26 epoch", TEAL), ("full_latest", "full · 300 epoch", RED)):
    with np.load(OUT / f"predictions_{label}.npz") as d:
        values = np.sort(d["mean_position_3d_error"])
    ax.plot(values, np.arange(1, len(values) + 1) / len(values), color=color, label=title)
ax.yaxis.set_major_formatter(PercentFormatter(1))
ax.set(xlabel="샘플 내 5전하 평균 3D 거리 오차", ylabel="누적 시험 샘플 비율", xlim=(0, 1.9), ylim=(0, 1.02), title="위치 오차 분포도 마지막 epoch에서 악화")
ax.legend(frameon=False, fontsize=9)
ax.grid()
ax = axes[1, 0]
rows = [r for r in strata if r["model"] == "full_best" and r["group"] == "true_potential_rms"]
bars = ax.bar(["낮은 25%", "25–50%", "50–75%", "높은 25%"], [float(r["mean_position_3d_error"]) for r in rows], color=["#a8ccca", "#7cb7b2", "#459d94", TEAL], width=.56)
ax.bar_label(bars, fmt="%.3f", padding=3)
ax.set(xlabel="정답 전위 RMS 사분위 (각 250개)", ylabel="평균 3D 거리 오차", ylim=(0, 1), title="신호가 약한 표본에서 위치 복원이 더 어려움")
ax.grid(axis="y")
ax.set_axisbelow(True)
ax = axes[1, 1]
for label, name, color in (("full_best", "full · 26 epoch", TEAL), ("full_latest", "full · 300 epoch", RED)):
    rows = [r for r in calibration if r["model"] == label]
    ax.plot([float(r["mean_confidence"]) for r in rows], [float(r["accuracy"]) for r in rows], "o-", label=name, color=color, ms=4)
ax.plot([.5, 1], [.5, 1], color=GRAY, ls="--", lw=1, label="보정된 예측")
ax.set(xlabel="전체부호 예측 확신도", ylabel="해당 구간의 실제 정확도", xlim=(.49, 1.01), ylim=(.2, 1.03), title="전체부호 보정: 일부 구간은 표본이 적음")
ax.legend(frameon=False, fontsize=9)
ax.grid()
save(fig, "prediction_diagnostics", "위치만 매칭은 기존 축별 정규화 위치 MSE로 계산. 서로 다른 평가 관점이며 기존 매칭이 잘못됐다는 뜻은 아님.")


with np.load(ROOT / "Models/charge_dataset_5charges_v9.npz") as d:
    gx, gy = np.meshgrid(d["grid_x"], d["grid_y"])
    targets = d["target"].reshape(-1, 5, 4)
with np.load(OUT / "predictions_full_best.npz") as d:
    pred = {k: d[k] for k in d.files}
with np.load(OUT / "predictions_full_latest.npz") as d:
    last = {k: d[k] for k in d.files}


def field(pos, q):
    distance = np.sqrt((gx[..., None] - pos[:, 0]) ** 2 + (gy[..., None] - pos[:, 1]) ** 2 + pos[:, 2] ** 2)
    return (q / (4 * np.pi * distance)).sum(axis=-1)


fig = plt.figure(figsize=(14, 10))
fig.subplots_adjust(top=.86, bottom=.08, hspace=.39, wspace=.28)
fig.suptitle("오차 분위수로 고른 실제 예측 예시", x=.055, y=.965, ha="left", fontsize=21, weight="bold")
fig.text(.055, .917, "샘플 평균 3D 거리 오차의 10·50·90% 지점에 가장 가까운 표본을 자동 선택 · 행마다 같은 전위 색 범위", color="#52637a", fontsize=10.5)
for row, quantile in enumerate((.1, .5, .9)):
    i = int(np.argmin(np.abs(pred["mean_position_3d_error"] - np.quantile(pred["mean_position_3d_error"], quantile))))
    target = targets[pred["test_indices"][i]]
    fs = [field(target[:, :3], target[:, 3]), field(pred["pred_position"][i], pred["pred_charge"][i]), field(last["pred_position"][i], last["pred_charge"][i])]
    limit = max(float(np.abs(f).max()) for f in fs)
    map_axes = []
    for col, title in enumerate(("정답 전위 V", "선택: 26 epoch", "마지막: 300 epoch")):
        ax = fig.add_subplot(3, 4, row * 4 + col + 1)
        map_axes.append(ax)
        im = ax.imshow(fs[col], origin="lower", extent=[-2, 2, -2, 2], cmap="coolwarm", vmin=-limit, vmax=limit)
        ax.set_title(title, fontsize=10)
        ax.set(xticks=[-2, 0, 2], yticks=[-2, 0, 2], xlabel="x")
        if col == 0:
            ax.set_ylabel(f"{int(quantile * 100)}% 분위 · ID {pred['test_indices'][i]}\ny")
        else:
            error = (pred if col == 1 else last)["field_relative_l2"][i]
            ax.text(.03, .04, f"전위 상대 L2: {error:.2f}", transform=ax.transAxes, color="black", fontsize=8,
                    bbox={"facecolor": "white", "alpha": .8, "edgecolor": "none", "pad": 3})
    colorbar = fig.colorbar(im, ax=map_axes, fraction=.018, pad=.016, shrink=.82)
    colorbar.set_label("V", fontsize=8)
    colorbar.ax.tick_params(labelsize=7)
    ax = fig.add_subplot(3, 4, row * 4 + 4, projection="3d")
    truth, predicted = pred["target_position"][i], pred["pred_position"][i]
    for point in range(5):
        joined = np.stack((truth[point], predicted[point]))
        ax.plot(joined[:, 0], joined[:, 1], joined[:, 2], color=GRAY, lw=.8, alpha=.7)
    ax.scatter(*truth.T, c=[RED if q > 0 else BLUE for q in pred["target_charge"][i]], marker="o", s=55, label="정답")
    ax.scatter(*predicted.T, c=[RED if q > 0 else BLUE for q in pred["pred_charge"][i]], marker="^", s=55, label="예측")
    ax.set(xlim=(-1.7, 1.7), ylim=(-1.7, 1.7), zlim=(-.2, 1.7), xlabel="x", ylabel="y", zlabel="z")
    ax.set_title(f"26 epoch 위치 · 오차 {pred['mean_position_3d_error'][i]:.3f}", fontsize=10)
    ax.tick_params(labelsize=7)
    ax.view_init(22, -54)
    if row == 0:
        ax.legend(loc="upper left", frameon=False, fontsize=8)
save(fig, "reconstruction_examples", "위치 그림: 원=정답, 삼각형=예측, 빨강=양전하, 파랑=음전하. 선은 기존 공동 매칭 결과. 단위는 생성 데이터의 임의 단위.")
print("Created four PNG and SVG research figures.")
