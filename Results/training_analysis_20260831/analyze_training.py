"""Read-only audit and inference analysis of the last completed five-charge sweep.

Run from the repository with .venv/Scripts/python.exe. No training, checkpoint
updates, report refreshes in experiment folders, or network access are performed.
Only this analysis directory receives new files. Bootstrap units are whole
five-charge test examples, never individual charges.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / "Codes"))
import ModelExperiment9 as experiment
import NewLearning9 as physics
import torch

RESULTS = ROOT / "Results/new_learning9_experiments"
MODELS = ROOT / "Models/new_learning9_experiments"
FRACTIONS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
N_BOOTSTRAP = 10000


def write_json(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def write_csv(name, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with (OUT / name).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def safe_checkpoint(path):
    # The latest snapshots include NumPy's RNG array. Permit only its known
    # array constructor and dtype, retaining the restricted Torch unpickler.
    with torch.serialization.safe_globals([
        np._core.multiarray._reconstruct, np.ndarray, np.dtype, np.dtypes.UInt32DType,
    ]):
        return torch.load(path, map_location="cpu", weights_only=True)


class StripDocstrings(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        if (node.body and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)):
            node.body = node.body[1:]
        return node

    visit_AsyncFunctionDef = visit_FunctionDef
    visit_ClassDef = visit_FunctionDef
    visit_Module = visit_FunctionDef


def descriptive(values):
    a = np.asarray(values, dtype=float)
    return {"mean": float(a.mean()), "median": float(np.median(a)),
            "p90": float(np.quantile(a, .9)), "p95": float(np.quantile(a, .95)),
            "max": float(a.max())}


def bootstrap_improvement(a, b, higher_is_better=False):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    delta = np.asarray(b) - np.asarray(a) if higher_is_better else np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(20260831)
    means = []
    for _ in range(N_BOOTSTRAP // 250):
        indices = rng.integers(0, len(delta), size=(250, len(delta)))
        means.extend(delta[indices].mean(axis=1))
    low, high = np.quantile(means, [.025, .975])
    return {"improvement_mean": float(delta.mean()), "ci95_low": float(low), "ci95_high": float(high),
            "sample_count": len(delta), "bootstrap_draws": N_BOOTSTRAP,
            "interpretation": "Conditional on these two fitted models and this test distribution; not seed uncertainty."}


def audit_records():
    records, rows, paired, histories, protocols = [], [], [], {}, {}
    checks = {"completed_runs": 0, "history_epochs": 0, "csv_metric_checks": 0,
              "best_epoch_checks": 0, "checkpoint_files_present": 0, "checkpoint_files_missing": 0}
    for directory in sorted(RESULTS.glob("5point_routing_v1_seed*")):
        protocol = json.loads((directory / "protocol.json").read_text(encoding="utf-8"))
        protocols[directory.name] = protocol
        payload = {k: v for k, v in protocol.items() if k != "protocol_fingerprint"}
        assert experiment.object_fingerprint(payload) == protocol["protocol_fingerprint"]
        saved_csv = list(csv.DictReader((directory / "runs.csv").open(encoding="utf-8-sig", newline="")))
        assert len(saved_csv) == 24
        for file in sorted(directory.glob("runs/*/result.json")):
            record = json.loads(file.read_text(encoding="utf-8"))
            experiment.completed_result_evaluations(record)
            status = json.loads(file.with_name("status.json").read_text(encoding="utf-8"))
            config = json.loads(file.with_name("config.json").read_text(encoding="utf-8"))
            assert status["status"] == "completed" and status["run_id"] == record["run_id"]
            assert config == record["configuration"]
            history = json.loads(file.with_name("history.json").read_text(encoding="utf-8"))
            assert len(history) == record["training_result"]["epochs_completed"] == 300
            assert [r["epoch"] for r in history] == list(range(1, 301))
            for h in history:
                for phase in ("train", "validation"):
                    experiment.validate_loss_values(h[phase], config, "audit")
            record["_directory"] = directory.name
            record["_source"] = str(file.relative_to(ROOT))
            records.append(record)
            histories[record["run_id"]] = history
            checks["completed_runs"] += 1
            checks["history_epochs"] += len(history)
            for selection, evaluation in record["evaluations"].items():
                best = min(history, key=lambda h: h["validation"][selection])
                assert best["epoch"] == evaluation["selected_epoch"]
                assert best["validation"] == evaluation["validation_losses"]
                checks["best_epoch_checks"] += 1
                csv_row = next(r for r in saved_csv if r["run_id"] == record["run_id"] and r["checkpoint_selection"] == selection)
                assert int(csv_row["selected_epoch"]) == evaluation["selected_epoch"]
                for metric, value in evaluation["test_metrics"].items():
                    assert (not csv_row[metric]) if value is None else float(csv_row[metric]) == value
                    checks["csv_metric_checks"] += 1
                rows.append({"seed": record["seed"], "protocol_fingerprint": record["protocol_fingerprint"],
                             "model": record["model_name"], "fraction": record["g05_fraction"],
                             "sensor_count": record["g05_count_per_sample"], "selection": selection,
                             "epoch": evaluation["selected_epoch"], "validation_loss": evaluation["selected_validation_loss"],
                             **evaluation["test_metrics"], "source": record["_source"]})
            for name in ("latest.pt", "best_structure.pt", "best_total.pt"):
                exists = (MODELS / directory.name / record["run_id"] / name).is_file()
                checks["checkpoint_files_present" if exists else "checkpoint_files_missing"] += 1
    checks["sign_only_structure_history_identical_across_fractions"] = {}
    checks["zero_sensor_ab_histories_and_test_metrics_identical"] = {}
    for seed in (41, 42, 43):
        subset = [r for r in records if r["seed"] == seed and r["model_name"] == "g05_sign_only"]
        first = histories[subset[0]["run_id"]]
        same = all(all(h[phase][metric] == first[i][phase][metric]
                       for i, h in enumerate(histories[r["run_id"]]) for phase in ("train", "validation")
                       for metric in ("structure", "position", "magnitude", "relative_sign")) for r in subset)
        checks["sign_only_structure_history_identical_across_fractions"][str(seed)] = same
        zero = [r for r in records if r["seed"] == seed and r["g05_fraction"] == 0]
        zero_equal = histories[zero[0]["run_id"]] == histories[zero[1]["run_id"]]
        zero_equal &= all(zero[0]["evaluations"][s]["test_metrics"] == zero[1]["evaluations"][s]["test_metrics"]
                          for s in ("structure", "total"))
        checks["zero_sensor_ab_histories_and_test_metrics_identical"][str(seed)] = zero_equal
        for fraction in FRACTIONS:
            a, b = [next(r for r in records if r["seed"] == seed and r["g05_fraction"] == fraction
                         and r["model_name"] == name) for name in experiment.DEFAULT_MODELS]
            assert a["protocol_fingerprint"] == b["protocol_fingerprint"]
            for selection in ("structure", "total"):
                for metric in experiment.METRIC_NAMES:
                    av, bv = (r["evaluations"][selection]["test_metrics"][metric] for r in (a, b))
                    improvement = None if av is None else (av - bv if metric in experiment.LOWER_IS_BETTER else bv - av)
                    paired.append({"seed": seed, "protocol_fingerprint": a["protocol_fingerprint"], "fraction": fraction,
                                   "sensor_count": a["g05_count_per_sample"], "selection": selection, "metric": metric,
                                   "sign_only": av, "full": bv, "improvement": improvement,
                                   "relative_improvement_percent": 100 * improvement / av if av and improvement is not None else None})
    write_csv("saved_metrics.csv", rows)
    write_csv("paired_comparisons.csv", paired)
    checks["distinct_protocols"] = len({p["protocol_fingerprint"] for p in protocols.values()})
    checks["split_indices_identical"] = len({experiment.canonical_json(p["physics"]["split_indices"]) for p in protocols.values()}) == 1
    checks["normalization_identical"] = len({experiment.canonical_json(p["normalization"]) for p in protocols.values()}) == 1
    old = subprocess.check_output(["git", "show", "ac59a6cc:Codes/NewLearning9.py"], cwd=ROOT)
    current = (ROOT / "Codes/NewLearning9.py").read_bytes()
    checks["baseline_executable_ast_equal_to_training_version"] = (
        ast.dump(StripDocstrings().visit(ast.parse(old))) == ast.dump(StripDocstrings().visit(ast.parse(current))))
    checks["baseline_training_hashes_match_lf_or_crlf"] = all(
        p["source_sha256"]["NewLearning9.py"] in (hashlib.sha256(old).hexdigest(), hashlib.sha256(old.replace(b"\n", b"\r\n")).hexdigest())
        for p in protocols.values())
    routing = (ROOT / "Codes/ModelExperiment9.py").read_bytes().replace(b"\r\n", b"\n")
    checks["routing_source_hashes_match_lf_or_crlf"] = all(p["source_sha256"]["ModelExperiment9.py"] in (
        hashlib.sha256(routing).hexdigest(), hashlib.sha256(routing.replace(b"\n", b"\r\n")).hexdigest()) for p in protocols.values())
    assert checks["baseline_executable_ast_equal_to_training_version"]
    assert all(checks["sign_only_structure_history_identical_across_fractions"].values())
    assert all(checks["zero_sensor_ab_histories_and_test_metrics_identical"].values())
    return records, histories, protocols, checks


@torch.inference_mode()
def collect_predictions(model, dataset, stats):
    device = next(model.parameters()).device
    chunks = defaultdict(list)
    for batch in physics.create_data_loader(dataset, 128, device=device):
        g00, g05, mask, position, charge = (t.to(device) for t in batch)
        output = model(g00, g05, mask)
        assignment = physics.minimum_cost_assignment(physics.matching_cost(output, position, charge))
        pos_target, q_target = physics.matched_targets(position, charge, assignment)
        geo_cost = (output.position[:, :, None] - position[:, None]).square().mean(dim=-1)
        geo_assignment = physics.minimum_cost_assignment(geo_cost)
        geo_position, geo_charge = physics.matched_targets(position, charge, geo_assignment)
        relative_target, global_target = physics.canonical_sign_targets(q_target)
        geo_relative, _ = physics.canonical_sign_targets(geo_charge)
        relative = physics.decode_relative_signs(output.relative_sign_logit)
        q_pred = physics.reconstruct_charges(output) * stats.charge_scale
        q_target = q_target * stats.charge_scale
        pos_std = torch.as_tensor(stats.position_std, device=device)
        pos_mean = torch.as_tensor(stats.position_mean, device=device)
        position_error = (output.position - pos_target) * pos_std
        geo_error = (output.position - geo_position) * pos_std
        correct_relative = relative == (relative_target * 2 - 1)
        correct_global = (output.global_sign_logit >= 0) == global_target.bool()
        correct_absolute = (q_pred > 0) == (q_target > 0)
        abs_error = position_error.abs()
        observed = mask.sum(dim=(1, 2)) > 0
        direct = (q_pred - q_target).abs().mean(dim=1)
        invariant = torch.minimum(direct, (-q_pred - q_target).abs().mean(dim=1))
        probability = torch.sigmoid(output.global_sign_logit)
        pattern_confidence = physics.relative_pattern_scores(output.relative_sign_logit).softmax(dim=1).max(dim=1).values
        global_oracle = output.magnitude * stats.charge_scale * relative * (global_target * 2 - 1)[:, None]
        values = {
            "mean_position_mae": abs_error.mean(dim=(1, 2)),
            "mean_position_3d_error": position_error.norm(dim=-1).mean(dim=1),
            "position_mae_x": abs_error[:, :, 0].mean(dim=1),
            "position_mae_y": abs_error[:, :, 1].mean(dim=1),
            "position_mae_z": abs_error[:, :, 2].mean(dim=1),
            "charge_magnitude_mae": (output.magnitude * stats.charge_scale - q_target.abs()).abs().mean(dim=1),
            "charge_mae": torch.where(observed, direct, invariant),
            "global_invariant_charge_mae": invariant,
            "relative_sign_accuracy": correct_relative.float().mean(dim=1),
            "relative_configuration_accuracy": correct_relative.all(dim=1).float(),
            "global_sign_accuracy": correct_global.float(),
            "absolute_sign_accuracy": correct_absolute.float().mean(dim=1),
            "absolute_sign_set_accuracy": correct_absolute.all(dim=1).float(),
            "geometric_relative_sign_accuracy": (relative == (geo_relative * 2 - 1)).float().mean(dim=1),
            "geometric_relative_configuration_accuracy": (relative == (geo_relative * 2 - 1)).all(dim=1).float(),
            "geometric_mean_position_3d_error": geo_error.norm(dim=-1).mean(dim=1),
            "relative_negative_count_accuracy": ((relative < 0).sum(dim=1) == (relative_target == 0).sum(dim=1)).float(),
            "global_probability": probability,
            "global_target": global_target,
            "global_confidence": torch.maximum(probability, 1 - probability),
            "relative_confidence": pattern_confidence,
            "global_oracle_charge_mae": (global_oracle - q_target).abs().mean(dim=1),
            "pred_position": output.position * pos_std + pos_mean,
            "pred_charge": q_pred,
            "target_position": pos_target * pos_std + pos_mean,
            "target_charge": q_target,
            "per_charge_3d_error": position_error.norm(dim=-1),
            "relative_sign": relative,
            "joint_assignment": assignment,
            "geometric_assignment": geo_assignment,
        }
        for name, value in values.items():
            chunks[name].append(value.cpu().numpy())
    return {name: np.concatenate(values) for name, values in chunks.items()}


def load_model(record, selection, device):
    name = "latest.pt" if selection == "latest" else f"best_{selection}.pt"
    checkpoint = safe_checkpoint(MODELS / record["_directory"] / record["run_id"] / name)
    if selection == "latest":
        experiment.validate_resume_checkpoint(checkpoint, record["configuration"])
    else:
        evaluation = record["evaluations"][selection]
        experiment.validate_selected_checkpoint(checkpoint, record["configuration"], selection=selection,
                                               expected_epoch=evaluation["selected_epoch"],
                                               expected_loss=evaluation["selected_validation_loss"])
    model = experiment.MODEL_REGISTRY[record["model_name"]].factory().to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, checkpoint


def evaluate_checkpoints(records, arrays, split, stats, device):
    reevaluations, predictions, max_discrepancy = [], {}, 0.0
    for record in [r for r in records if r["seed"] == 43]:
        dataset = physics.prepare_dataset(arrays, split.test, stats, record["g05_fraction"])
        for selection in ("structure", "total", "latest"):
            model, checkpoint = load_model(record, selection, device)
            with torch.inference_mode():
                metrics = physics.evaluate_model(model, dataset, stats, batch_size=128)
            epoch = checkpoint["epoch"] if selection == "latest" else checkpoint["selected_epoch"]
            row = {"seed": 43, "model": record["model_name"], "fraction": record["g05_fraction"],
                   "selection": selection, "epoch": epoch, **metrics}
            if selection != "latest":
                saved = record["evaluations"][selection]["test_metrics"]
                difference = max(abs(value - saved[key]) for key, value in metrics.items() if value is not None)
                max_discrepancy = max(max_discrepancy, difference)
                row["max_abs_difference_from_saved"] = difference
                assert difference < 1e-6, (record["run_id"], selection, difference)
            reevaluations.append(row)
            # Whole-example predictions support paired inference and error strata.
            if selection == "structure" or (record["g05_fraction"] == 1 and selection == "latest"):
                per = collect_predictions(model, dataset, stats)
                for key in ("mean_position_mae", "mean_position_3d_error", "charge_magnitude_mae", "charge_mae"):
                    assert abs(float(per[key].mean(dtype=np.float64)) - metrics[key]) < 1e-6
                predictions[(record["model_name"], record["g05_fraction"], selection)] = per
            del model, checkpoint
        print(f"Verified selected and latest checkpoints: {record['run_id']}", flush=True)
    write_csv("reevaluated_metrics_seed43.csv", reevaluations)
    return reevaluations, predictions, max_discrepancy


def potential(positions, charges, grid_x, grid_y, epsilon_0):
    gx, gy = np.meshgrid(grid_x, grid_y)
    outputs = []
    for start in range(0, len(positions), 128):
        pos = np.asarray(positions[start:start + 128], dtype=np.float64)
        q = np.asarray(charges[start:start + 128], dtype=np.float64)
        distances = np.sqrt((gx.ravel()[None, :, None] - pos[:, None, :, 0]) ** 2
                            + (gy.ravel()[None, :, None] - pos[:, None, :, 1]) ** 2 + pos[:, None, :, 2] ** 2)
        outputs.append((q[:, None] / (4 * np.pi * epsilon_0 * np.maximum(distances, 1e-12))).sum(axis=-1))
    return np.concatenate(outputs)


def dataset_audit(arrays, split, protocols, stats):
    path = ROOT / "Models/charge_dataset_5charges_v9.npz"
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    p43 = protocols["5point_routing_v1_seed43"]
    assert sha == p43["data"]["sha256"]
    for protocol in protocols.values():
        indices = protocol["physics"]["split_indices"]
        union = np.concatenate([indices[k] for k in ("train", "validation", "test")])
        assert len(union) == len(np.unique(union)) == len(arrays.target)
    recalculated = physics.calculate_normalization_stats(arrays, split.train).to_dict()
    assert recalculated == p43["normalization"]
    field = potential(arrays.target[:, :, :3], arrays.target[:, :, 3], arrays.grid_x, arrays.grid_y, arrays.epsilon_0)
    g00 = arrays.g00.reshape(len(field), -1)
    x, y = arrays.g05[0, :, :2].astype(int).T
    g05 = field[:, y * len(arrays.grid_x) + x]
    consistent_g00 = np.isclose(field ** 2, g00, rtol=2e-4, atol=2e-6)
    consistent_g05 = np.isclose(g05, arrays.g05[:, :, 2], rtol=2e-4, atol=2e-6)
    signs = np.sign(arrays.target[split.test, :, 3])
    relative = signs * signs.prod(axis=1)[:, None]
    audit = {"sha256": sha, "matches_seed43": True,
             "matches_seed41_42_file_hash": sha == protocols["5point_routing_v1_seed42"]["data"]["sha256"],
             "all_10000_examples_g00_physics_pass": bool(consistent_g00.all()),
             "all_10000_examples_g05_physics_pass": bool(consistent_g05.all()),
             "max_absolute_g00_physics_residual": float(np.max(np.abs(field ** 2 - g00))),
             "max_absolute_g05_physics_residual": float(np.max(np.abs(g05 - arrays.g05[:, :, 2]))),
             "target_duplicate_count": len(arrays.target) - len(np.unique(arrays.target.reshape(len(arrays.target), -1), axis=0)),
             "train_normalization_recomputed_exactly": True,
             "test_global_positive_fraction": float((signs.prod(axis=1) > 0).mean()),
             "test_relative_negative_count_distribution": {str(k): int(((relative < 0).sum(axis=1) == k).sum()) for k in (0, 2, 4)},
             "test_coordinate_ranges": {axis: [float(arrays.target[split.test, :, i].min()), float(arrays.target[split.test, :, i].max())]
                                        for i, axis in enumerate(("x", "y", "z"))}}
    assert audit["all_10000_examples_g00_physics_pass"] and audit["all_10000_examples_g05_physics_pass"]
    return audit, field[split.test]


def analyze_predictions(predictions, arrays, split, true_field):
    a = predictions[("g05_sign_only", 1.0, "structure")]
    b = predictions[("g05_full_reconstruction", 1.0, "structure")]
    last = predictions[("g05_full_reconstruction", 1.0, "latest")]
    paired = []
    for fraction in FRACTIONS:
        aa = predictions[("g05_sign_only", fraction, "structure")]
        bb = predictions[("g05_full_reconstruction", fraction, "structure")]
        metrics = ["mean_position_mae", "mean_position_3d_error", "charge_magnitude_mae", "relative_configuration_accuracy"]
        if fraction:
            metrics += ["global_sign_accuracy", "absolute_sign_set_accuracy", "charge_mae"]
        for key in metrics:
            paired.append({"fraction": fraction, "metric": key,
                           **bootstrap_improvement(aa[key], bb[key], key.endswith("accuracy"))})
    write_csv("paired_bootstrap_seed43.csv", paired)
    metrics = ("mean_position_mae", "mean_position_3d_error", "charge_magnitude_mae", "charge_mae",
               "global_invariant_charge_mae", "global_oracle_charge_mae", "relative_sign_accuracy",
               "relative_configuration_accuracy", "global_sign_accuracy", "absolute_sign_set_accuracy",
               "geometric_relative_sign_accuracy", "geometric_relative_configuration_accuracy",
               "geometric_mean_position_3d_error", "relative_negative_count_accuracy")
    diagnostic = {}
    reliability, strata = [], []
    original_targets = arrays.target[split.test]
    target_distance = np.linalg.norm(original_targets[:, :, None, :3] - original_targets[:, None, :, :3], axis=-1)
    target_distance[:, np.arange(5), np.arange(5)] = np.inf
    minimum_separation = target_distance.min(axis=(1, 2))
    true_rms = np.sqrt(np.mean(true_field ** 2, axis=1))
    for label, per in (("sign_only_best", a), ("full_best", b), ("full_latest", last)):
        predicted_field = potential(per["pred_position"], per["pred_charge"], arrays.grid_x, arrays.grid_y, arrays.epsilon_0)
        per["field_relative_l2"] = np.sqrt(np.mean((predicted_field - true_field) ** 2, axis=1)) / np.maximum(true_rms, 1e-12)
        per["magnitude_field_relative_l2"] = np.sqrt(np.mean((np.abs(predicted_field) - np.abs(true_field)) ** 2, axis=1)) / np.maximum(true_rms, 1e-12)
        true_g00 = true_field ** 2
        per["g00_relative_l2"] = np.sqrt(np.mean((predicted_field ** 2 - true_g00) ** 2, axis=1)) / np.maximum(np.sqrt(np.mean(true_g00 ** 2, axis=1)), 1e-12)
        values = {key: descriptive(per[key]) for key in metrics}
        values.update({key: descriptive(per[key]) for key in ("field_relative_l2", "magnitude_field_relative_l2", "g00_relative_l2")})
        values["per_charge_3d_error"] = descriptive(per["per_charge_3d_error"].ravel())
        values["fraction_charges_with_3d_error_below_0p25"] = float((per["per_charge_3d_error"] < .25).mean())
        values["fraction_charges_with_3d_error_below_0p50"] = float((per["per_charge_3d_error"] < .5).mean())
        values["fraction_all_five_3d_errors_below_0p50"] = float((per["per_charge_3d_error"] < .5).all(axis=1).mean())
        values["z_nonpositive_fraction"] = float((per["pred_position"][:, :, 2] <= 0).mean())
        values["position_outside_generation_support_fraction"] = float(((np.abs(per["pred_position"][:, :, :2]) > 1.5 + 1e-6).any(axis=-1)
            | (per["pred_position"][:, :, 2] < .1 - 1e-6) | (per["pred_position"][:, :, 2] > 1.5 + 1e-6)).mean())
        values["charge_magnitude_outside_generation_support_fraction"] = float(((np.abs(per["pred_charge"]) < .3 - 1e-6)
            | (np.abs(per["pred_charge"]) > 1 + 1e-6)).mean())
        values["global_relative_correctness_counts"] = {f"global_{g}_relative_{r}": int(((per["global_sign_accuracy"] == g)
            & (per["relative_configuration_accuracy"] == r)).sum()) for g in (0, 1) for r in (0, 1)}
        confidence = per["global_confidence"]
        ece = 0.0
        for lo in np.arange(.5, 1, .05):
            mask = (confidence >= lo) & (confidence < lo + .05 + (1e-6 if lo > .94 else 0))
            if not mask.any():
                continue
            accuracy = float(per["global_sign_accuracy"][mask].mean())
            mean_conf = float(confidence[mask].mean())
            ece += mask.mean() * abs(accuracy - mean_conf)
            reliability.append({"model": label, "confidence_lower": float(lo), "count": int(mask.sum()),
                                "mean_confidence": mean_conf, "accuracy": accuracy})
        values["global_ece_10_confidence_bins"] = float(ece)
        values["global_brier_score"] = float(np.mean((per["global_probability"] - per["global_target"]) ** 2))
        values["relative_prediction_confidence_mean"] = float(per["relative_confidence"].mean())
        values["relative_wrong_prediction_confidence_mean"] = float(per["relative_confidence"][per["relative_configuration_accuracy"] == 0].mean())
        values["relative_configuration_equals_count_correct_for_all_samples"] = bool(np.array_equal(
            per["relative_configuration_accuracy"], per["relative_negative_count_accuracy"]))
        for group_name, group_values, cutoffs in (
            ("minimum_charge_separation", minimum_separation, np.quantile(minimum_separation, [0, .25, .5, .75, 1])),
            ("true_potential_rms", true_rms, np.quantile(true_rms, [0, .25, .5, .75, 1])),
            ("mean_charge_height", original_targets[:, :, 2].mean(axis=1), np.quantile(original_targets[:, :, 2].mean(axis=1), [0, .25, .5, .75, 1])),
        ):
            for i in range(4):
                mask = (group_values >= cutoffs[i]) & (group_values <= cutoffs[i + 1] if i == 3 else group_values < cutoffs[i + 1])
                strata.append({"model": label, "group": group_name, "quartile": i + 1, "lower": float(cutoffs[i]),
                               "upper": float(cutoffs[i + 1]), "count": int(mask.sum()),
                               **{key: float(per[key][mask].mean()) for key in ("mean_position_3d_error", "charge_mae", "global_sign_accuracy", "relative_configuration_accuracy", "field_relative_l2")}})
        diagnostic[label] = values
        np.savez_compressed(OUT / f"predictions_{label}.npz", test_indices=split.test, **per)
        one_dimensional = [key for key, value in per.items() if value.ndim == 1]
        write_csv(f"per_example_{label}.csv", [{"test_example_index": int(split.test[i]),
                  **{key: float(per[key][i]) for key in one_dimensional}} for i in range(len(split.test))])
    write_csv("calibration.csv", reliability)
    write_csv("error_strata.csv", strata)
    write_json("prediction_diagnostics.json", diagnostic)
    return diagnostic, paired, strata, reliability


def simulate_early_stopping(records, histories):
    rows = []
    for patience in (10, 15, 20, 30):
        for record in records:
            history = histories[record["run_id"]]
            best = {"structure": math.inf, "total": math.inf}
            last_improved, stop = 0, 300
            for h in history:
                changed = False
                for selection in best:
                    if h["validation"][selection] < best[selection]:
                        best[selection] = h["validation"][selection]
                        changed = True
                if changed:
                    last_improved = h["epoch"]
                if h["epoch"] - last_improved >= patience:
                    stop = h["epoch"]
                    break
            preserved = all(best[s] == record["training_result"][f"best_{s}_loss"] for s in best)
            rows.append({"run_id": record["run_id"], "patience": patience, "stop_epoch": stop,
                         "both_best_checkpoints_preserved": preserved, "epochs_saved": 300 - stop})
    write_csv("early_stopping_replay.csv", rows)
    return {str(p): {"both_bests_preserved_run_count": sum(r["both_best_checkpoints_preserved"] for r in rows if r["patience"] == p),
                    "total_stop_epochs": sum(r["stop_epoch"] for r in rows if r["patience"] == p),
                    "epoch_reduction_percent": 100 * sum(r["epochs_saved"] for r in rows if r["patience"] == p) / (36 * 300),
                    "stop_epoch_min": min(r["stop_epoch"] for r in rows if r["patience"] == p),
                    "stop_epoch_max": max(r["stop_epoch"] for r in rows if r["patience"] == p)} for p in (10, 15, 20, 30)}


@torch.inference_mode()
def intervention_diagnostics(record, arrays, split, stats, device):
    model, _ = load_model(record, "structure", device)
    dataset = physics.prepare_dataset(arrays, split.test, stats, 1.0)
    base = collect_predictions(model, dataset, stats)
    model.allow_g05_for_structure = False
    removed = collect_predictions(model, dataset, stats)
    model.allow_g05_for_structure = True
    rows = []
    selected = ("mean_position_mae", "mean_position_3d_error", "charge_magnitude_mae", "relative_configuration_accuracy", "global_sign_accuracy", "charge_mae")
    for name, per in (("original_full", base), ("same_full_weights_context_removed", removed)):
        rows.append({"intervention": name, **{key: float(per[key].mean(dtype=np.float64)) for key in selected}})
    ratio, maximum_symmetry_error = [], {key: 0.0 for key in experiment.OUTPUT_FIELDS}
    for batch in physics.create_data_loader(dataset, 128, device=device):
        g00, g05, mask, _, _ = (t.to(device) for t in batch)
        structure = model.g00_encoder(model.g00_cnn(g00))
        reversed_g05 = g05 * g05.new_tensor((1, 1, -1))
        summary = model._masked_summary(model.g05_encoder(torch.cat((g05, reversed_g05))), torch.cat((mask, mask)))
        plus, minus = summary.chunk(2)
        context = model.structure_context((plus + minus) * .5)
        ratio.extend((context.norm(dim=1) / structure.norm(dim=1).clamp_min(1e-12)).cpu().tolist())
        original, flipped = model(g00, g05, mask), model(g00, reversed_g05, mask)
        for key in experiment.OUTPUT_FIELDS:
            left, right = getattr(original, key), getattr(flipped, key)
            residual = left + right if key == "global_sign_logit" else left - right
            maximum_symmetry_error[key] = max(maximum_symmetry_error[key], float(residual.abs().max()))
    write_csv("context_ablation_seed43_full100.csv", rows)
    return {"context_to_g00_feature_l2_ratio": descriptive(ratio), "max_sign_reversal_residual": maximum_symmetry_error,
            "interventions": rows, "caution": "Removing context after fitting is a distribution shift, not a retrained ablation."}


def main():
    print("Auditing saved records, histories, protocols and CSVs...", flush=True)
    records, histories, protocols, checks = audit_records()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    physics.set_reproducibility(43)
    torch.set_num_threads(10)
    torch.set_num_interop_threads(10)
    p43 = protocols["5point_routing_v1_seed43"]
    torch.backends.cuda.matmul.allow_tf32 = p43["environment"]["cuda_matmul_allow_tf32"]
    torch.backends.cudnn.allow_tf32 = p43["environment"]["cudnn_allow_tf32"]
    torch.set_float32_matmul_precision(p43["environment"]["float32_matmul_precision"])
    arrays = physics.load_dataset(ROOT / "Models/charge_dataset_5charges_v9.npz")
    indices = p43["physics"]["split_indices"]
    split = physics.DataSplit(**{k: np.asarray(v, dtype=np.int64) for k, v in indices.items()})
    stats = experiment.normalization_from_config(next(r for r in records if r["seed"] == 43)["configuration"])
    print("Checking all 10,000 physical examples and split integrity...", flush=True)
    data_audit, true_field = dataset_audit(arrays, split, protocols, stats)
    print("Reevaluating seed43 checkpoints with the original 1,000 test examples...", flush=True)
    reevaluations, predictions, max_discrepancy = evaluate_checkpoints(records, arrays, split, stats, device)
    print("Computing whole-example paired bootstrap, assignment sensitivity and field reconstruction...", flush=True)
    diagnostic, paired, strata, calibration = analyze_predictions(predictions, arrays, split, true_field)
    latest = max(records, key=lambda r: r["completed_at"])
    interventions = intervention_diagnostics(latest, arrays, split, stats, device)
    best_epochs = [r["training_result"]["best_structure_epoch"] for r in records]
    total_epochs = [r["training_result"]["best_total_epoch"] for r in records]
    summary = {"last_completed_run": {k: latest[k] for k in ("run_id", "completed_at", "seed", "g05_fraction", "model_name", "training_result")},
               "audit": checks, "dataset": data_audit,
               "runtime": {"python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda, "device": str(device)},
               "reevaluation_max_abs_metric_difference": max_discrepancy,
               "best_structure_epoch_range": [min(best_epochs), max(best_epochs)],
               "best_total_epoch_range": [min(total_epochs), max(total_epochs)],
               "sum_logged_training_elapsed_minutes": sum(r["training_result"]["elapsed_seconds"] for r in records) / 60,
               "early_stopping_replay": simulate_early_stopping(records, histories),
               "last_run_history_selected": histories[latest["run_id"]][latest["training_result"]["best_structure_epoch"] - 1],
               "last_run_history_final": histories[latest["run_id"]][-1],
               "interventions": interventions,
               "scope_limits": ["Seed41 and seed42 checkpoints are not present locally.",
                                "Seed41/42 NPZ file hash differs from seed43; array identity cannot be verified without the old archive.",
                                "Runtime differs across seeds; cross-seed results are descriptive, not one controlled replicated protocol.",
                                "Bootstrap intervals condition on already fitted models; no multiple-comparison adjustment.",
                                "Latest checkpoint evaluation is a post-hoc diagnostic; it does not affect stored model selection."]}
    write_json("audit_summary.json", summary)
    print(json.dumps({"audit": checks, "maximum_saved_metric_difference": max_discrepancy,
                      "latest_full_diagnostics": diagnostic["full_best"],
                      "early_stopping": summary["early_stopping_replay"], "interventions": interventions}, ensure_ascii=False, indent=2), flush=True)
    print(f"Analysis files: {OUT}", flush=True)


if __name__ == "__main__":
    main()
