"""Read-only 3D visualization of validation-selected Experiment11 checkpoints.

The two-charge visualize_3d.py is intentionally unchanged. This entry point
uses ModelExperiment11.load_trained_model and NewLearning9's exact joint
assignment, not position-only matching or the old lexicographic slot order.
Only trusted, locally generated PyTorch checkpoints may be loaded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

import ModelExperiment10 as routing
import ModelExperiment11 as experiment
import NewLearning9 as physics


ROOT = Path(__file__).resolve().parents[1]
SWEEP_SCHEMA = "model-experiment11-finalized-fraction-seed-v1"
SCHEMA = "experiment11-3d-visualization-v2"
PAIRING_POLICY = (
    "NewLearning9.minimum_cost_assignment(matching_cost(output, target, saved_loss_weights)); "
    "one joint position/magnitude/relative-sign assignment for all fields; "
    "no G05/global-sign cost"
)
SAMPLE_POLICY = (
    "Illustrative samples from the entire saved test split, ranked by mean five-charge "
    "3D position error AFTER the evaluation's joint assignment. "
    "No checkpoint/model/hyperparameter selection uses these errors."
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_json(path: Path) -> dict[str, Any]:
    value = experiment.read_json(path)
    require(isinstance(value, dict), f"Expected a JSON object: {path}")
    return value


@dataclass(frozen=True)
class RuntimeContext:
    checkpoint_path: Path
    checkpoint: dict[str, Any]
    study_dir: Path
    model: routing.RoutedChargeNet
    stats: physics.NormalizationStats
    arrays: physics.DatasetArrays
    split: physics.DataSplit
    provenance: dict[str, Any]

    @property
    def config(self) -> dict[str, Any]:
        return self.checkpoint["configuration"]

    @property
    def seed(self) -> int:
        return int(self.config["training"]["seed"])

    @property
    def fraction(self) -> float:
        return float(self.config["observation"]["g05_fraction"])

    @property
    def selection(self) -> str:
        return self.checkpoint["checkpoint_selection"]


@dataclass(frozen=True)
class InferenceBatch:
    inference_batch_size: int
    dataset_indices: np.ndarray
    positions: np.ndarray
    raw_charges: np.ndarray
    target_positions: np.ndarray
    target_charges: np.ndarray
    assignment: np.ndarray  # predicted slot -> original target slot, zero based
    masks: np.ndarray
    position_errors: np.ndarray
    global_sign_logits: np.ndarray
    per_sample_metrics: dict[str, np.ndarray]

    @property
    def errors(self) -> np.ndarray:
        return self.per_sample_metrics["mean_position_3d_error"]


def find_authority(checkpoint_path: Path, study_dir: Path | None) -> tuple[Path, dict[str, Any]]:
    directories = (study_dir.expanduser().resolve(),) if study_dir is not None else checkpoint_path.parents
    for directory in directories:
        for filename in ("protocol.json", "study.json"):
            path = directory / filename
            if not path.is_file():
                continue
            document = read_json(path)
            if document.get("schema") in (experiment.SCHEMA, SWEEP_SCHEMA):
                experiment.unseal(document)
                return path, document
    raise FileNotFoundError(
        "No supported Experiment11 study.json or finalized-seed protocol.json found. "
        "Use --study-dir with the complete original study; a .pt file alone has no test split."
    )


def source_verification(study_dir: Path, expected_hashes: dict[str, str]) -> dict[str, Any]:
    """Permit Git LF/CRLF conversion only; never dynamically execute snapshots."""
    records = {}
    for module in (routing, experiment, physics):
        current = Path(module.__file__).resolve()
        expected = expected_hashes.get(current.name)
        require(isinstance(expected, str), f"Missing saved source hash: {current.name}")
        snapshot = study_dir / "sources" / current.name
        current_bytes = current.read_bytes()
        saved_bytes = snapshot.read_bytes() if snapshot.is_file() else current_bytes
        lf = saved_bytes.replace(b"\r\n", b"\n")
        variants = (saved_bytes, lf, lf.replace(b"\n", b"\r\n"))
        require(expected in {hashlib.sha256(value).hexdigest() for value in variants},
                f"Source snapshot/hash mismatch: {current.name}")
        require(current_bytes.replace(b"\r\n", b"\n") == lf,
                f"Inference source changed: {current.name}; restore matching source, do not guess")
        records[current.name] = {
            "recorded_sha256": expected,
            "current_sha256": hashlib.sha256(current_bytes).hexdigest(),
            "snapshot_sha256": hashlib.sha256(saved_bytes).hexdigest() if snapshot.is_file() else None,
            "match": "exact" if hashlib.sha256(current_bytes).hexdigest() == expected else "lf_crlf_only",
        }
    return records


def verify_selected_trial(
    study_dir: Path, document: dict[str, Any], checkpoint: dict[str, Any], checkpoint_path: Path,
) -> dict[str, Any]:
    """Read only THIS trial's metadata/weights, never other seeds' checkpoints."""
    config = checkpoint["configuration"]
    directory = experiment.trial_dir(study_dir, config).resolve()
    require(directory.parent == (study_dir / "runs").resolve(), "Invalid trial path")
    require(read_json(directory / "config.json") == config, "Checkpoint and saved config.json differ")
    record = experiment.read_trial(study_dir, config, verify_checkpoints=False)
    selection = checkpoint["checkpoint_selection"]
    selected = record["evaluations"][selection]
    digest = routing.file_sha256(checkpoint_path)
    require(digest == selected["checkpoint_sha256"], "Checkpoint differs from validation-selected weights")
    for key in ("selected_epoch", "selected_validation_loss", "validation_losses"):
        require(checkpoint[key] == selected[key], f"Checkpoint and validation record differ: {key}")

    is_sweep = document["schema"] == SWEEP_SCHEMA
    lock_path = study_dir / ("validation_selection.json" if is_sweep else "selection.json")
    require(lock_path.is_file(), "Test access denied: validation selection has not been locked")
    lock = read_json(lock_path)
    experiment.unseal(lock)
    require(lock.get("schema") == document["schema"], "Unexpected selection-lock schema")
    require(lock.get("test_used_for_selection") is False, "Selection lock is not validation-only")
    if is_sweep:
        require(lock.get("protocol_identity_fingerprint") == experiment.digest(document["identity"]),
                "Validation lock belongs to another seed/protocol")
        require(lock.get("seed") == config["training"]["seed"], "Validation-lock seed mismatch")
        entries = lock["runs"]
    else:
        require(lock.get("study_fingerprint") == document["fingerprint"],
                "Selection lock belongs to another study")
        entries = lock["evaluation_runs"]
    matches = [item for item in entries if item.get("configuration") == config and item.get("trial") == directory.name]
    require(len(matches) == 1, "Checkpoint is not one of the validation-locked evaluation runs")
    relative_checkpoint = (directory / f"best_{selection}.pt").relative_to(study_dir).as_posix()
    require(selected["checkpoint_path"] == relative_checkpoint, "Validation checkpoint path mismatch")
    artifact_hashes = lock["artifact_sha256"]
    verified = {}
    for relative, actual in (
        (relative_checkpoint, digest),
        ((directory / "result.json").relative_to(study_dir).as_posix(), routing.file_sha256(directory / "result.json")),
        ((directory / "history.json").relative_to(study_dir).as_posix(), routing.file_sha256(directory / "history.json")),
    ):
        require(artifact_hashes.get(relative) == actual, f"Validation-locked artifact changed: {relative}")
        verified[relative] = actual
    return {
        "checkpoint_sha256": digest,
        "selection_lock": str(lock_path), "selection_lock_fingerprint": lock["fingerprint"],
        "validation_record_fingerprint": record["fingerprint"], "validation": selected,
        "verified_artifacts": verified,
        "verification_scope": "requested trial and checkpoint only; no other seed weights/results loaded",
    }


def restore_runtime_context(
    checkpoint_path: Path, data_override: Path | None = None,
    device: torch.device = torch.device("cpu"), study_dir: Path | None = None,
) -> RuntimeContext:
    checkpoint_path = checkpoint_path.expanduser().resolve()
    require(checkpoint_path.is_file(), f"Checkpoint does not exist: {checkpoint_path}")
    before_hash = routing.file_sha256(checkpoint_path)
    # Native restoration includes saved routing, dropout, full-state shape and
    # selected-epoch checks. Its temporary initialization must not affect caller RNG.
    with torch.random.fork_rng(devices=[]):
        model, stats, checkpoint = experiment.load_trained_model(checkpoint_path, device)
    config = checkpoint["configuration"]
    require(config.get("tuning", {}).get("schema") == experiment.SCHEMA,
            "This loader requires an Experiment11 checkpoint, not the old two-charge/baseline model")
    authority_path, document = find_authority(checkpoint_path, study_dir)
    study_dir = authority_path.parent
    if document["schema"] == SWEEP_SCHEMA:
        identity = document["identity"]
        require(experiment.digest(config) in identity["run_configuration_fingerprints"],
                "Checkpoint is absent from the frozen per-seed configurations")
        require(config["tuning"]["study_fingerprint"] == identity["parent_study_fingerprint"],
                "Checkpoint belongs to another parent protocol")
        require(config["training"]["seed"] == identity["seed"], "Checkpoint seed differs from isolated study")
        require(config["model"]["name"] in identity["models"]
                and config["observation"]["g05_fraction"] in identity["fractions"], "Condition not declared")
        data_metadata = identity["data"]
        expected_sources = {name: row["current_sha256"] for name, row in identity["source_audit"].items()}
        environment = identity["environment"]
        split_seed = int(data_metadata["split_seed"])
    else:
        require(config["tuning"]["study_fingerprint"] == document["fingerprint"],
                "Checkpoint belongs to another Experiment11 study")
        candidate = next((row for row in document["candidates"] if row["id"] == config["tuning"]["candidate_id"]), None)
        require(candidate is not None, "Checkpoint candidate is absent from the saved study")
        expected = experiment.configuration(document, candidate, config["model"]["name"],
                                            config["observation"]["g05_fraction"], config["training"]["seed"])
        require(config == expected, "Checkpoint configuration differs from the original study")
        require(config["normalization"] == document["normalization"], "Saved normalization mismatch")
        data_metadata, expected_sources = document["data"], document["source_sha256"]
        environment, split_seed = document["environment"], int(document["split_seed"])

    # Verify the completed validation minimum and test gate BEFORE opening data.
    provenance = verify_selected_trial(study_dir, document, checkpoint, checkpoint_path)
    require(provenance["checkpoint_sha256"] == before_hash, "Checkpoint changed during loading")
    sources = source_verification(study_dir, expected_sources)
    data_path = routing.resolve_evaluation_dataset({"data": data_metadata}, data_override)
    arrays = physics.load_dataset(data_path)
    require(experiment.dataset_hashes(arrays) == data_metadata["array_hashes"], "Dataset array hashes differ")
    split_path = study_dir / "split_indices.npz"
    names = ("train", "validation", "test")
    if document["schema"] == SWEEP_SCHEMA:
        require(split_path.is_file(), "Missing saved split_indices.npz; inference never invents a split")
        with np.load(split_path, allow_pickle=False) as saved:
            indices = {name: saved[name].copy() for name in names}
        # This standalone manifest records the fixed split seed rather than full
        # index lists. Audit the archive against that algorithm; USE the archive.
        expected_split = physics.create_data_split(len(arrays.target), seed=split_seed)
        require(all(np.array_equal(indices[name], getattr(expected_split, name)) for name in names),
                "Saved split differs from the frozen data-split seed")
    else:
        indices = {name: np.asarray(document["split"][name]) for name in names}
        if split_path.is_file():
            with np.load(split_path, allow_pickle=False) as saved:
                require(all(np.array_equal(saved[name], indices[name]) for name in names), "Saved split archive changed")
    require(all(value.ndim == 1 and value.dtype.kind in "iu" and len(value) == config["split_counts"][name]
                and len(value) > 0 for name, value in indices.items()), "Invalid saved split sizes/dtypes")
    require(np.array_equal(np.sort(np.concatenate(list(indices.values()))), np.arange(len(arrays.target))),
            "Saved splits must be disjoint and cover the original dataset exactly")
    require(read_json(study_dir / "normalization.json") == config["normalization"], "Train-only normalization changed")
    for name in ("position_mean", "position_std"):
        value = getattr(stats, name)
        require(value.shape == (3,) and bool(np.isfinite(value).all()), f"Invalid five-charge normalization: {name}")
    require(np.isfinite(stats.g00_mean) and bool((stats.position_std > 0).all()) and all(np.isfinite(value) and value > 0
            for value in (stats.g00_std, stats.g05_value_scale, stats.charge_scale)), "Invalid normalization scales")
    observation = config["observation"]
    require(observation["candidate_count"] == arrays.g05.shape[1]
            and observation["g05_count_per_sample"] == physics.g05_count_for_fraction(
                observation["g05_fraction"], arrays.g05.shape[1]), "Saved G05 fraction/mask policy differs")
    provenance.update({
        "schema": SCHEMA, "authority_path": str(authority_path), "authority_fingerprint": document["fingerprint"],
        "authority_schema": document["schema"], "data_path": str(data_path), "data_sha256": data_metadata["sha256"],
        "split_seed": split_seed, "split_counts": config["split_counts"],
        "split_index_hashes": {name: experiment.array_hash(value) for name, value in indices.items()},
        "normalization": config["normalization"], "sources": sources, "training_environment": environment,
        "matching": PAIRING_POLICY, "sample_selection_policy": SAMPLE_POLICY,
        "test_set": "original saved test split (historical test for the tuning study), NOT fresh_test",
        "no_training_or_selection_changes": True,
    })
    return RuntimeContext(checkpoint_path, checkpoint, study_dir, model, stats, arrays,
                          physics.DataSplit(**indices), provenance)


@torch.inference_mode()
def infer_test(context: RuntimeContext, batch_size: int | None = None) -> InferenceBatch:
    batch_size = context.config["training"]["batch_size"] if batch_size is None else batch_size
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("Inference batch size must be a positive integer")
    model, stats = context.model, context.stats
    model.eval()
    device = next(model.parameters()).device
    weights = physics.LossWeights(**context.config["training"]["loss_weights"])
    dataset = physics.prepare_dataset(context.arrays, context.split.test, stats, context.fraction)
    outputs: dict[str, list[np.ndarray]] = {}

    def append(name: str, tensor: torch.Tensor) -> None:
        require(bool(torch.isfinite(tensor).all()), f"Non-finite inference output: {name}")
        outputs.setdefault(name, []).append(tensor.cpu().numpy())

    position_std = torch.as_tensor(stats.position_std, device=device)
    position_mean = torch.as_tensor(stats.position_mean, device=device)
    for tensors in physics.create_data_loader(dataset, batch_size, device=device):
        g00, g05, mask, position, charge = (tensor.to(device) for tensor in tensors)
        output = model(g00, g05, mask)
        # This is EXACTLY the evaluator's assignment, in normalized loss units.
        # Sorting by XYZ or matching only positions would change the experiment.
        assignment = physics.minimum_cost_assignment(physics.matching_cost(output, position, charge, weights))
        position, charge = physics.matched_targets(position, charge, assignment)
        predicted_q = physics.reconstruct_charges(output) * stats.charge_scale
        target_q = charge * stats.charge_scale
        position_error = (output.position - position) * position_std
        errors = position_error.norm(dim=-1)
        direct = (predicted_q - target_q).abs().mean(dim=1)
        invariant = torch.minimum(direct, (-predicted_q - target_q).abs().mean(dim=1))
        observed = mask.sum(dim=(1, 2)) > 0
        for name, value in {
            "positions": output.position * position_std + position_mean,
            "raw_charges": predicted_q,
            "target_positions": position * position_std + position_mean,
            "target_charges": target_q, "assignment": assignment, "masks": mask[:, :, 0],
            "position_errors": errors, "global_sign_logits": output.global_sign_logit,
            "mean_position_3d_error": errors.mean(dim=1),
            "mean_position_mae": position_error.abs().mean(dim=(1, 2)),
            "charge_mae": torch.where(observed, direct, invariant),
            "global_invariant_charge_mae": invariant,
            "charge_magnitude_mae": (output.magnitude * stats.charge_scale - target_q.abs()).abs().mean(dim=1),
        }.items():
            append(name, value)
    combined = {key: np.concatenate(values) for key, values in outputs.items()}
    metric_names = ("mean_position_3d_error", "mean_position_mae", "charge_mae",
                    "global_invariant_charge_mae", "charge_magnitude_mae")
    metrics = {name: combined.pop(name) for name in metric_names}
    batch = InferenceBatch(inference_batch_size=batch_size, dataset_indices=context.split.test.copy(),
                           per_sample_metrics=metrics, **combined)
    require(batch.positions.shape == (len(dataset), physics.CHARGE_COUNT, 3), "Unexpected five-charge output shape")
    matched_raw = context.arrays.target[batch.dataset_indices][np.arange(len(dataset))[:, None], batch.assignment]
    require(np.allclose(batch.target_positions, matched_raw[:, :, :3], rtol=2e-6, atol=2e-6)
            and np.allclose(batch.target_charges, matched_raw[:, :, 3], rtol=2e-6, atol=2e-6),
            "Matched target denormalization differs from the original data")
    count = context.config["observation"]["g05_count_per_sample"]
    require(bool((batch.masks[:, :count] == 1).all() and (batch.masks[:, count:] == 0).all()),
            "Inference masks differ from the saved nested sensor prefix")
    return batch


def select_samples(batch: InferenceBatch, mode: str, sample_index: int | None = None) -> list[tuple[str, int]]:
    if sample_index is not None:
        if not 0 <= sample_index < len(batch.errors):
            raise ValueError(f"--sample-index must be in [0, {len(batch.errors) - 1}] within the saved test split")
        return [("index", sample_index)]
    # A stable tie break makes the selected original test index reproducible.
    order = np.argsort(batch.errors, kind="stable")
    positions = {"best": 0, "median": (len(order) - 1) // 2, "worst": len(order) - 1}
    modes = ("best", "median", "worst") if mode == "all" else (mode,)
    return [(name, int(order[positions[name]])) for name in modes]


def sample_record(context: RuntimeContext, batch: InferenceBatch, mode: str, index: int) -> dict[str, Any]:
    observed = bool(batch.masks[index].any())
    return {
        "mode": mode, "test_index": index, "dataset_index": int(batch.dataset_indices[index]),
        "test_count": len(batch.errors), "error_rank": int(np.count_nonzero(batch.errors < batch.errors[index])) + 1,
        "error_percentile": float(np.mean(batch.errors <= batch.errors[index]) * 100),
        "assignment_predicted_to_original_target_0based": batch.assignment[index].tolist(),
        "predicted_positions": batch.positions[index].tolist(), "raw_predicted_charges": batch.raw_charges[index].tolist(),
        "matched_true_positions": batch.target_positions[index].tolist(), "matched_true_charges": batch.target_charges[index].tolist(),
        "per_charge_3d_errors": batch.position_errors[index].tolist(),
        "metrics": {name: float(value[index]) for name, value in batch.per_sample_metrics.items()},
        "g05_mask": batch.masks[index].astype(bool).tolist(),
        "global_sign_logit": float(batch.global_sign_logits[index]),
        "absolute_sign_identifiable": observed,
        "prediction_sign_representation": "absolute prediction" if observed else "whole-vector +/- equivalence class",
        "global_sign_accuracy": None if not observed else bool(
            (batch.global_sign_logits[index] >= 0) == (np.prod(np.sign(batch.target_charges[index])) > 0)),
        "absolute_sign_accuracy": None if not observed else float(np.mean(
            (batch.raw_charges[index] > 0) == (batch.target_charges[index] > 0))),
        "oracle_sign_used_for_plot": False,
    }


def create_output_dir(base: Path, context: RuntimeContext) -> Path:
    # Unique leaf + mkdir(exist_ok=False): no seed/model/selection/older image is overwritten.
    path = (base.expanduser().resolve() / f"seed{context.seed}" / context.config["model"]["name"]
            / f"g{context.fraction:g}" / f"{context.selection}_{context.provenance['checkpoint_sha256'][:12]}"
            / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:8]}")
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_json_exclusive(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def inference_environment(device: torch.device) -> dict[str, Any]:
    # The training helper's environment() hardcodes its deterministic policy.
    # Library callers may use different flags; report actual inference settings.
    result = routing.runtime_environment(device)
    result.update(deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                  deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
                  cudnn_benchmark=torch.backends.cudnn.benchmark,
                  cudnn_deterministic=torch.backends.cudnn.deterministic)
    return result


def render_plot(
    context: RuntimeContext, batch: InferenceBatch, sample: dict[str, Any], output: Path,
    *, show_g00: bool = False, show_g05: bool = False, dpi: int = 180, show_window: bool = False,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D

    index = sample["test_index"]
    true, predicted = batch.target_positions[index], batch.positions[index]
    true_q, predicted_q = batch.target_charges[index], batch.raw_charges[index]
    observed = sample["absolute_sign_identifiable"]
    figure = plt.figure(figsize=(15, 9), facecolor="white")
    axis = figure.add_axes((0.02, 0.12, 0.63, 0.77), projection="3d")
    panel = figure.add_axes((0.69, 0.10, 0.29, 0.77))
    panel.axis("off")
    figure.suptitle(f"Experiment11 | {context.config['model']['name']} | {sample['mode']} test example",
                   fontsize=16, x=0.5, y=0.97)
    figure.text(0.5, 0.925,
                "G05=0: absolute/global sign is unidentifiable. Gray predictions represent one whole-vector +/- class."
                if not observed else "Five-charge reconstruction with the evaluation's joint permutation-invariant matching",
                ha="center", fontsize=10, color="#77532a" if not observed else "#4c5763")
    x, y = np.meshgrid(context.arrays.grid_x, context.arrays.grid_y)
    if show_g00:
        g00 = context.arrays.g00[sample["dataset_index"]]
        scale = Normalize(float(context.arrays.g00.min()), float(context.arrays.g00.max()))
        color = plt.get_cmap("viridis")
        axis.plot_surface(x, y, np.zeros_like(x), facecolors=color(scale(g00)),
                          shade=False, alpha=0.45, rstride=1, cstride=1, linewidth=0)
        color_axis = figure.add_axes((0.09, 0.095, 0.47, 0.017))
        colorbar = figure.colorbar(plt.cm.ScalarMappable(norm=scale, cmap=color), cax=color_axis, orientation="horizontal")
        colorbar.set_label("G00 = V squared (original dataset-wide scale)", fontsize=9)
        colorbar.ax.tick_params(labelsize=8)
    else:
        axis.plot_surface(x, y, np.zeros_like(x), color="#afbdcc", shade=False, alpha=0.15, linewidth=0)
    if show_g05 and observed:
        sensors = context.arrays.g05[sample["dataset_index"]][batch.masks[index].astype(bool)]
        axis.scatter(context.arrays.grid_x[np.rint(sensors[:, 0]).astype(int)],
                     context.arrays.grid_y[np.rint(sensors[:, 1]).astype(int)], np.zeros(len(sensors)),
                     marker="s", c="#263746", s=20, depthshade=False)
    all_positions = np.vstack((true, predicted))
    bounds = ((min(context.arrays.grid_x.min(), all_positions[:, 0].min()),
               max(context.arrays.grid_x.max(), all_positions[:, 0].max())),
              (min(context.arrays.grid_y.min(), all_positions[:, 1].min()),
               max(context.arrays.grid_y.max(), all_positions[:, 1].max())),
              (min(0., all_positions[:, 2].min()), max(context.arrays.target[:, :, 2].max(), all_positions[:, 2].max())))
    limits = [(float(low - .04 * max(high - low, 1e-6)), float(high + .04 * max(high - low, 1e-6))) for low, high in bounds]
    def color_for(charge: float) -> str:
        return "#c3473f" if charge > 0 else "#326fad"

    for slot in range(physics.CHARGE_COUNT):
        line = np.vstack((true[slot], predicted[slot]))
        axis.plot(*line.T, color="#596879", linewidth=1.4, alpha=.8)
        axis.scatter(*true[slot], marker="o", c=color_for(true_q[slot]), s=95,
                     edgecolors="white", linewidths=.8, depthshade=False)
        axis.scatter(*predicted[slot], marker="x", c=color_for(predicted_q[slot]) if observed else "#737b84",
                     s=100, linewidths=2.4, depthshade=False)
        dz = .025 * (limits[2][1] - limits[2][0])
        axis.text(*(true[slot] + np.array([0., 0., dz])), f"T{batch.assignment[index, slot] + 1}", fontsize=9)
        axis.text(*(predicted[slot] - np.array([0., 0., dz])), f"P{slot + 1}", fontsize=9)
    axis.set(xlim=limits[0], ylim=limits[1], zlim=limits[2], xlabel="x", ylabel="y", zlabel="z")
    axis.set_box_aspect([high - low for low, high in limits])
    axis.view_init(elev=25, azim=-55)
    axis.legend(handles=[
        Line2D([], [], marker="o", color="none", markerfacecolor="#596879", label="True charge (T)", markersize=8),
        Line2D([], [], marker="x", color="#596879", linestyle="none", label="Prediction (P)", markersize=8),
        Line2D([], [], color="#596879", label="Joint-matched position error"),
    ], loc="upper left", fontsize=9, framealpha=.9)

    count = context.config["observation"]["g05_count_per_sample"]
    candidate_count = context.config["observation"]["candidate_count"]
    panel.text(0, 1, "FROZEN MODEL / SAVED TEST SPLIT", fontsize=11, weight="bold", va="top", color="#283b50")
    summary = (
        f"Seed {context.seed}   |   G05 {context.fraction:g} ({count}/{candidate_count})\n"
        f"Checkpoint: best_{context.selection}   |   epoch {context.checkpoint['selected_epoch']}\n"
        f"Validation {context.selection}: {context.checkpoint['selected_validation_loss']:.6f}\n"
        f"Checkpoint hash: {context.provenance['checkpoint_sha256'][:12]}\n\n"
        f"Test index {index}   |   original row {sample['dataset_index']}\n"
        f"Illustrative rank: {sample['error_rank']}/{sample['test_count']}\n"
        f"Mean 3D error: {sample['metrics']['mean_position_3d_error']:.6f}\n"
        f"Full test mean / median: {np.mean(batch.errors):.6f} / {np.median(batch.errors):.6f}"
    )
    panel.text(0, .94, summary, va="top", fontsize=10, linespacing=1.6)
    rows = [[f"P{slot + 1} / T{batch.assignment[index, slot] + 1}", f"{true_q[slot]:+.4f}",
             f"{predicted_q[slot]:+.4f}", f"{batch.position_errors[index, slot]:.4f}"] for slot in range(physics.CHARGE_COUNT)]
    table = panel.table(cellText=rows, colLabels=("Matched pair", "True q", "Pred q*" if not observed else "Pred q", "3D error"),
                        colWidths=(.31, .23, .23, .23), bbox=(0, .33, 1, .26), cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#dbe2e8")
        cell.set_facecolor("#eaf0f5" if row == 0 else ("#f7f9fb" if row % 2 else "white"))
    note = (
        "* Pred q is one representative of +/- q.\nFlip ALL FIVE signs together, never individually.\n"
        "Absolute/global sign accuracy: N/A.\nNo oracle sign is used to color predictions."
        if not observed else "Charge colors: red (+), blue (-).\nPrediction signs are raw model outputs;\nno target-based sign flip is applied."
    )
    panel.text(0, .28, note, va="top", fontsize=9, linespacing=1.45, color="#536375")
    panel.text(0, .11, "T = original target slot, not a physical charge ID.\nP = predicted slot. One joint assignment is shared\nby positions, magnitudes and relative signs.",
               va="top", fontsize=9, linespacing=1.45, color="#536375")
    figure.text(.035, .025, "Post-training illustration only. Best/median/worst refer to samples, never checkpoint selection.",
                fontsize=9, color="#536375")
    try:
        with output.open("xb") as handle:
            figure.savefig(handle, format="png", dpi=dpi, facecolor="white")
        if show_window:
            plt.show()
    finally:
        plt.close(figure)


def export_visualizations(
    context: RuntimeContext, batch: InferenceBatch, *, output_dir: Path, mode: str = "median",
    sample_index: int | None = None, show_g00: bool = False, show_g05: bool = False,
    dpi: int = 180, show_window: bool = False,
) -> Path:
    selections = select_samples(batch, mode, sample_index)
    directory = create_output_dir(output_dir, context)
    files = []
    for name, index in selections:
        sample = sample_record(context, batch, name, index)
        stem = f"{name}_seed{context.seed}_test{index:04d}_data{sample['dataset_index']:05d}"
        png = directory / f"{stem}.png"
        render_plot(context, batch, sample, png, show_g00=show_g00, show_g05=show_g05, dpi=dpi, show_window=show_window)
        write_json_exclusive(directory / f"{stem}.json", {
            "schema": SCHEMA, "checkpoint": str(context.checkpoint_path), "configuration": context.config,
            "provenance": context.provenance, "sample": sample,
        })
        files.append({"png": png.name, "sample_metadata": f"{stem}.json", "sha256": routing.file_sha256(png)})
        print(f"Saved PNG: {png}", flush=True)
    predictions = directory / f"predictions_seed{context.seed}.npz"
    with predictions.open("xb") as handle:
        np.savez_compressed(handle,
            **{name: value for name, value in vars(batch).items() if name != "per_sample_metrics"},
            **{f"metric_{name}": value for name, value in batch.per_sample_metrics.items()})
    import matplotlib
    write_json_exclusive(directory / "manifest.json", {
        "schema": SCHEMA, "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete", "checkpoint": str(context.checkpoint_path), "configuration": context.config,
        "provenance": context.provenance, "inference_environment": inference_environment(next(context.model.parameters()).device),
        "visualizer_sha256": routing.file_sha256(Path(__file__)), "matplotlib": matplotlib.__version__,
        "options": {"mode": mode, "sample_index": sample_index, "show_g00": show_g00, "show_g05": show_g05,
                    "dpi": dpi, "inference_batch_size": batch.inference_batch_size},
        "test_sample_count": len(batch.errors),
        "displayed_metric_means": {name: float(np.mean(values, dtype=np.float64)) for name, values in batch.per_sample_metrics.items()},
        "metrics_note": "A subset of unchanged evaluator metrics for visualization; does not replace final experiment results.",
        "figures": files, "predictions": predictions.name, "predictions_sha256": routing.file_sha256(predictions),
    })
    return directory


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Trusted Experiment11 best_structure.pt or best_total.pt")
    parser.add_argument("--study-dir", type=Path, help="Original study metadata, if the checkpoint was moved separately")
    parser.add_argument("--data", type=Path, help="Byte-identical relocated original dataset; SHA-256 must match")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--sample-index", type=int, help="Zero-based index WITHIN the saved test split")
    selection.add_argument("--sample-mode", choices=("best", "median", "worst", "all"), default="median")
    parser.add_argument("--show-g00", action="store_true")
    parser.add_argument("--show-g05", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "Results" / "visualizations_v2")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--batch-size", type=int, help="Inference only; defaults to the checkpoint's saved batch size")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--no-show", action="store_true", help="Save PNG/JSON/NPZ without opening a window")
    parser.add_argument("--check-only", action="store_true", help="Validate checkpoint/provenance/data without inference or output files")
    args = parser.parse_args(argv)
    if args.sample_index is not None and args.sample_index < 0:
        parser.error("--sample-index must be nonnegative")
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if not 72 <= args.dpi <= 600:
        parser.error("--dpi must be between 72 and 600")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    context = restore_runtime_context(args.checkpoint, args.data, device, args.study_dir)
    print(f"VERIFIED seed={context.seed} model={context.config['model']['name']} fraction={context.fraction:g} "
          f"best_{context.selection} epoch={context.checkpoint['selected_epoch']} test={len(context.split.test)}", flush=True)
    if args.check_only:
        return 0
    if args.sample_index is not None and args.sample_index >= len(context.split.test):
        raise ValueError(f"--sample-index must be smaller than {len(context.split.test)}")
    experiment.strict_seed(context.seed)
    environment = context.provenance["training_environment"]
    torch.backends.cuda.matmul.allow_tf32 = environment["cuda_matmul_allow_tf32"]
    torch.backends.cudnn.allow_tf32 = environment["cudnn_allow_tf32"]
    torch.set_num_threads(environment["torch_num_threads"])
    if args.no_show:
        import matplotlib
        matplotlib.use("Agg", force=True)
    batch = infer_test(context, args.batch_size)
    directory = export_visualizations(context, batch, output_dir=args.output_dir, mode=args.sample_mode,
                                     sample_index=args.sample_index, show_g00=args.show_g00, show_g05=args.show_g05,
                                     dpi=args.dpi, show_window=not args.no_show)
    print(f"Saved manifest: {directory / 'manifest.json'}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, FileNotFoundError) as error:
        print(f"Visualization refused: {error}", file=sys.stderr)
        raise SystemExit(1) from error
