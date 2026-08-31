"""Run one isolated seed for the finalized Experiment11 six-fraction protocol.

The finalized hyperparameters are read only from the sealed Experiment11
selection.  This runner never loads a prior run result or checkpoint.  It
trains with train/validation data only, freezes all validation-selected
checkpoints, and only then opens the existing test split for final evaluation.

Example (from the repository root)::

    python Codes/run_finalized_fraction_sweep.py \
        --seed 43 \
        --study-dir Modelexperiment11/studies/final_fraction_sweep_seed43_20260831 \
        --device cuda

Rerunning the same command resumes unfinished training from its own latest
checkpoints, or verifies completed artifacts without retraining or retesting.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import os
import shutil
import sys
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import ModelExperiment10 as legacy
import ModelExperiment11 as experiment
import NewLearning9 as physics
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARENT_STUDY = ROOT / "Modelexperiment11" / "studies" / "main"
DEFAULT_DATA = ROOT / "Models" / "charge_dataset_5charges_v9.npz"
DEFAULT_STUDY = ROOT / "Modelexperiment11" / "studies" / "final_fraction_sweep_seed43_20260831"
FRACTIONS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
MODELS = tuple(legacy.DEFAULT_MODELS)
CHECKPOINT_SELECTIONS = tuple(legacy.CHECKPOINT_SELECTIONS)
PRIMARY_SELECTION = "structure"
SCHEMA = "model-experiment11-finalized-fraction-seed-v1"
PARENT_SOURCES = tuple(experiment.SOURCES)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def normalized_source_bytes(path: Path) -> bytes:
    """Compare source code across harmless Git LF/CRLF conversion."""
    return path.read_bytes().replace(b"\r\n", b"\n")


def source_match_mode(current: Path, snapshot: Path) -> str:
    if current.read_bytes() == snapshot.read_bytes():
        return "exact"
    if normalized_source_bytes(current) == normalized_source_bytes(snapshot):
        return "lf_crlf_only"
    raise RuntimeError(f"Executable source differs from finalized snapshot: {current.name}")


def copy_snapshot(source: Path, destination: Path) -> None:
    """Atomically preserve the exact live source used by this seed run."""
    expected = legacy.file_sha256(source)
    if destination.exists():
        require(legacy.file_sha256(destination) == expected, f"Source snapshot changed: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with source.open("rb") as input_handle, temporary.open("wb") as output_handle:
            shutil.copyfileobj(input_handle, output_handle)
            output_handle.flush()
            os.fsync(output_handle.fileno())
        legacy.replace_with_retry(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    require(legacy.file_sha256(destination) == expected, f"Could not snapshot source: {source}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=43, help="One independent uint32 training seed")
    parser.add_argument("--study-dir", type=Path, default=DEFAULT_STUDY,
                        help="Dedicated output directory; it must include the seed in its name")
    parser.add_argument("--parent-study", type=Path, default=DEFAULT_PARENT_STUDY,
                        help="Sealed Experiment11 study containing the finalized selection")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--check-only", action="store_true",
                        help="Validate code/data/frozen settings without creating artifacts or training")
    args = parser.parse_args(argv)
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be a uint32 integer")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    return args


def resolve_device(name: str) -> torch.device:
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else "cpu") if name == "auto" else torch.device(name)


def ensure_isolated_output(study_dir: Path, seed: int, parent_study: Path) -> None:
    studies_root = (ROOT / "Modelexperiment11" / "studies").resolve()
    require(study_dir != parent_study, "The finalized parent study is read-only; use a new sibling directory")
    require(study_dir != studies_root, "Use a dedicated child directory for this seed")
    require(studies_root in study_dir.parents, f"Output must be below {studies_root}")
    require(f"seed{seed}" in study_dir.name.casefold(),
            f"Output directory name must contain seed{seed}: {study_dir.name}")
    protected = {ROOT / name for name in ("Codes", "Models", "Results", "Documents", ".git", ".venv")}
    require(not any(path == study_dir or path in study_dir.parents for path in protected),
            "Output must not be inside a code, model, result, or environment directory")


def configure_runtime(parent: dict[str, Any], device: torch.device) -> dict[str, Any]:
    expected = parent["environment"]
    for key in ("torch_num_threads", "torch_num_interop_threads"):
        require(type(expected.get(key)) is int and expected[key] > 0, f"Missing finalized {key}")
    # These calls occur before models/loaders are constructed in this new process.
    torch.set_num_threads(expected["torch_num_threads"])
    torch.set_num_interop_threads(expected["torch_num_interop_threads"])
    experiment.strict_seed(physics.DATA_SPLIT_SEED)
    return experiment.environment(device)


def load_parent(parent_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    study_path = parent_dir / "study.json"
    selection_path = parent_dir / "selection.json"
    require(study_path.is_file() and selection_path.is_file(), "Finalized parent study/selection is missing")
    parent = experiment.read_json(study_path)
    selection = experiment.read_json(selection_path)
    experiment.unseal(parent)
    experiment.unseal(selection)
    require(parent.get("schema") == experiment.SCHEMA, "Unsupported parent-study schema")
    experiment.validate_spec(parent["spec"])
    require(tuple(parent["spec"]["models"]) == MODELS, "Finalized model pair changed")
    require(selection.get("study_fingerprint") == parent["fingerprint"], "Selection belongs to another study")
    require(selection.get("test_used_for_selection") is False, "Finalized hyperparameters were not validation-only")
    candidate_id = selection.get("selected_candidate_id")
    candidate = next((item for item in parent["candidates"] if item["id"] == candidate_id), None)
    require(candidate is not None, "Selected candidate is absent from the parent study")
    require(selection.get("selected_hyperparameters") == candidate,
            "Selection hyperparameters differ from the declared candidate")
    require(parent["candidates"] == experiment.make_candidates(parent["spec"]),
            "Parent candidate table no longer matches its frozen search definition")
    return parent, selection, candidate


def audit_sources(parent_dir: Path, parent: dict[str, Any]) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for name in PARENT_SOURCES:
        current = ROOT / "Codes" / name
        snapshot = parent_dir / "sources" / name
        require(current.is_file() and snapshot.is_file(), f"Required source is missing: {name}")
        records[name] = {
            "parent_recorded_sha256": parent["source_sha256"][name],
            "parent_snapshot_sha256": legacy.file_sha256(snapshot),
            "current_sha256": legacy.file_sha256(current),
            "current_vs_parent_snapshot": source_match_mode(current, snapshot),
        }
    runner = Path(__file__).resolve()
    records[runner.name] = {
        "current_sha256": legacy.file_sha256(runner),
        "current_vs_parent_snapshot": "new_seed_isolation_runner",
    }
    return records


def verify_data(parent: dict[str, Any], data_path: Path) -> tuple[physics.DatasetArrays, physics.DataSplit, physics.NormalizationStats, dict[str, Any]]:
    require(data_path.is_file(), f"Dataset is missing: {data_path}")
    require(legacy.file_sha256(data_path) == parent["data"]["sha256"], "Dataset file hash differs from finalized study")
    arrays = physics.load_dataset(data_path)
    require(experiment.dataset_hashes(arrays) == parent["data"]["array_hashes"],
            "Dataset array hash differs from finalized study")
    split = physics.create_data_split(len(arrays.target), physics.DATA_SPLIT_SEED)
    for phase in ("train", "validation", "test"):
        require(np.array_equal(getattr(split, phase), np.asarray(parent["split"][phase])),
                f"Frozen {phase} split cannot be reproduced")
    stats = physics.calculate_normalization_stats(arrays, split.train)
    require(stats.to_dict() == parent["normalization"], "Train-only normalization differs from finalized study")
    return arrays, split, stats, {
        "path": str(data_path.resolve()),
        "sha256": parent["data"]["sha256"],
        "array_hashes": parent["data"]["array_hashes"],
        "split_seed": physics.DATA_SPLIT_SEED,
        "split_counts": {phase: len(getattr(split, phase)) for phase in ("train", "validation", "test")},
        "split_reproduced": True,
        "normalization_reproduced_from_train_only": True,
    }


def run_configurations(parent: dict[str, Any], candidate: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    configurations = [
        experiment.configuration(parent, candidate, model, fraction, seed)
        for fraction in FRACTIONS
        for model in MODELS
    ]
    require(len(configurations) == len(FRACTIONS) * len(MODELS), "Unexpected run count")
    for config, (fraction, model) in zip(configurations, ((f, m) for f in FRACTIONS for m in MODELS), strict=True):
        require(config["model"]["name"] == model, "Model order changed")
        require(config["training"]["seed"] == seed, "Seed changed in configuration")
        require(config["observation"]["g05_fraction"] == fraction, "Fraction changed in configuration")
        require(config["training"]["learning_rate"] == candidate["learning_rate"], "Learning rate changed")
        require(config["training"]["weight_decay"] == candidate["weight_decay"], "Weight decay changed")
        require(config["training"]["regularization"]["structure_dropout"] == candidate["structure_dropout"],
                "Structure dropout changed")
    return configurations


def immutable_or_same(path: Path, value: Any) -> None:
    experiment.immutable_json(path, value)


def ensure_split_archive(study_dir: Path, split: physics.DataSplit) -> None:
    path = study_dir / "split_indices.npz"
    expected = {phase: getattr(split, phase) for phase in ("train", "validation", "test")}
    if path.exists():
        with np.load(path, allow_pickle=False) as archive:
            require(set(archive.files) == set(expected), "Saved split archive fields changed")
            require(all(np.array_equal(archive[name], value) for name, value in expected.items()),
                    "Saved split archive differs from the frozen split")
    else:
        legacy.atomic_save_npz(path, **expected)


def initialize_output(
    study_dir: Path,
    *,
    parent_dir: Path,
    parent: dict[str, Any],
    selection: dict[str, Any],
    candidate: dict[str, Any],
    data_audit: dict[str, Any],
    source_audit: dict[str, Any],
    environment: dict[str, Any],
    configs: list[dict[str, Any]],
    split: physics.DataSplit,
    stats: physics.NormalizationStats,
    seed: int,
) -> dict[str, Any]:
    identity = {
        "parent_study_path": str(parent_dir),
        "parent_study_fingerprint": parent["fingerprint"],
        "parent_selection_fingerprint": selection["fingerprint"],
        "selected_candidate": candidate,
        "seed": seed,
        "fractions": list(FRACTIONS),
        "models": list(MODELS),
        "primary_checkpoint_selection": PRIMARY_SELECTION,
        "secondary_checkpoint_selection": "total (reported, never selected using test)",
        "data": data_audit,
        "source_audit": source_audit,
        "environment": environment,
        "independence": {
            "parent_checkpoints_loaded": False,
            "parent_trial_results_loaded": False,
            "prior_seed_weights_reused": False,
            "prior_seed_metrics_used_for_model_or_checkpoint_selection": False,
            "frozen_hyperparameter_authority": "parent selection metadata only",
            "output_root_is_seed_specific": True,
        },
        "protocol": {
            "training_input": "frozen train split only",
            "checkpoint_selection": "validation minima of whole-epoch states; first epoch breaks ties",
            "validation_primary": "best_structure.pt",
            "test_gate": "all 12 validation trials and checkpoint hashes are sealed before test TensorDataset construction",
            "test_used_for_model_selection": False,
            "loss_and_matching": "inherited unchanged from sealed ModelExperiment11/10 configuration",
        },
        "run_configuration_fingerprints": [experiment.digest(config) for config in configs],
    }
    protocol_path = study_dir / "protocol.json"
    if protocol_path.exists():
        protocol = experiment.read_json(protocol_path)
        experiment.unseal(protocol)
        require(protocol.get("schema") == SCHEMA and protocol.get("identity") == identity,
                "Existing output has a different seed/protocol identity")
    else:
        require(not study_dir.exists() or not any(study_dir.iterdir()),
                f"Refusing to write a new protocol into nonempty directory: {study_dir}")
        immutable_or_same(protocol_path, experiment.seal({
            "schema": SCHEMA,
            "created_at": legacy.utc_now(),
            "identity": identity,
        }))
    for name in PARENT_SOURCES:
        copy_snapshot(ROOT / "Codes" / name, study_dir / "sources" / name)
    copy_snapshot(Path(__file__).resolve(), study_dir / "sources" / Path(__file__).name)
    immutable_or_same(study_dir / "normalization.json", stats.to_dict())
    ensure_split_archive(study_dir, split)
    immutable_or_same(study_dir / "run_configurations.json", {
        "seed": seed,
        "fractions": list(FRACTIONS),
        "models": list(MODELS),
        "configurations": configs,
    })
    return identity


def development_data(
    arrays: physics.DatasetArrays,
    split: physics.DataSplit,
    stats: physics.NormalizationStats,
    fraction: float,
) -> experiment.DevelopmentData:
    return experiment.DevelopmentData(
        physics.prepare_dataset(arrays, split.train, stats, fraction),
        physics.prepare_dataset(arrays, split.validation, stats, fraction),
    )


def flat_validation_row(config: dict[str, Any], record: dict[str, Any], objective: str) -> dict[str, Any]:
    evaluation = record["evaluations"][objective]
    return {
        "trial": experiment.trial_dir(Path("."), config).name,
        "model": config["model"]["name"],
        "fraction": config["observation"]["g05_fraction"],
        "g05_count_per_sample": config["observation"]["g05_count_per_sample"],
        "seed": config["training"]["seed"],
        "checkpoint_selection": objective,
        "selected_epoch": evaluation["selected_epoch"],
        "selected_validation_loss": evaluation["selected_validation_loss"],
        "epochs_completed": record["training_result"]["epochs_completed"],
        "stop_reason": record["training_result"]["stop_reason"],
        "checkpoint_sha256": evaluation["checkpoint_sha256"],
        **{f"validation_loss_{name}": value for name, value in evaluation["validation_losses"].items()},
        **{f"validation_{name}": value for name, value in evaluation["validation_metrics"].items()},
    }


def validate_all_trials(study_dir: Path, configs: Iterable[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    records: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for config in configs:
        record = experiment.read_trial(study_dir, config, verify_checkpoints=True)
        require(record["test_evaluated"] is False, "A training trial reports test evaluation")
        records.append((config, record))
    return records


def make_validation_lock(
    study_dir: Path,
    *,
    identity: dict[str, Any],
    configs: list[dict[str, Any]],
) -> dict[str, Any]:
    path = study_dir / "validation_selection.json"
    if path.exists():
        return validate_validation_lock(study_dir, identity=identity, configs=configs)
    require(not (study_dir / "final_evaluation_started.json").exists(),
            "Cannot lock validation after test evaluation has started")
    trials = validate_all_trials(study_dir, configs)
    artifacts: dict[str, str] = {}
    runs: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    for config, record in trials:
        directory = experiment.trial_dir(study_dir, config)
        for filename in ("config.json", "history.json", "history.csv", "result.json", "latest.pt"):
            artifact = directory / filename
            require(artifact.is_file(), f"Missing validation artifact: {artifact}")
            artifacts[artifact.relative_to(study_dir).as_posix()] = legacy.file_sha256(artifact)
        checkpoints: dict[str, Any] = {}
        for objective in CHECKPOINT_SELECTIONS:
            evaluation = record["evaluations"][objective]
            checkpoint = study_dir / evaluation["checkpoint_path"]
            require(checkpoint.is_file(), f"Missing selected checkpoint: {checkpoint}")
            require(legacy.file_sha256(checkpoint) == evaluation["checkpoint_sha256"],
                    "Selected checkpoint hash changed before test access")
            artifacts[evaluation["checkpoint_path"]] = evaluation["checkpoint_sha256"]
            checkpoints[objective] = {
                "selected_epoch": evaluation["selected_epoch"],
                "selected_validation_loss": evaluation["selected_validation_loss"],
                "validation_losses": evaluation["validation_losses"],
                "checkpoint_path": evaluation["checkpoint_path"],
                "checkpoint_sha256": evaluation["checkpoint_sha256"],
            }
            validation_rows.append(flat_validation_row(config, record, objective))
        runs.append({
            "trial": directory.name,
            "configuration": config,
            "primary_checkpoint": checkpoints[PRIMARY_SELECTION],
            "checkpoints": checkpoints,
            "test_evaluated_during_training": False,
        })
    lock = experiment.seal({
        "schema": SCHEMA,
        "locked_at": legacy.utc_now(),
        "protocol_identity_fingerprint": experiment.digest(identity),
        "run_count": len(runs),
        "seed": identity["seed"],
        "fractions": identity["fractions"],
        "models": identity["models"],
        "primary_checkpoint_selection": PRIMARY_SELECTION,
        "test_used_for_selection": False,
        "runs": runs,
        "artifact_sha256": artifacts,
    })
    immutable_or_same(path, lock)
    legacy.atomic_write_csv(study_dir / "validation_results.csv", validation_rows)
    return lock


def validate_validation_lock(
    study_dir: Path,
    *,
    identity: dict[str, Any],
    configs: list[dict[str, Any]],
) -> dict[str, Any]:
    path = study_dir / "validation_selection.json"
    require(path.is_file(), "Test access denied: validation_selection.json is absent")
    lock = experiment.read_json(path)
    experiment.unseal(lock)
    require(lock.get("schema") == SCHEMA, "Unexpected validation-lock schema")
    require(lock.get("protocol_identity_fingerprint") == experiment.digest(identity),
            "Validation lock belongs to another protocol")
    require(lock.get("test_used_for_selection") is False, "Validation lock used test data")
    require(lock.get("primary_checkpoint_selection") == PRIMARY_SELECTION, "Primary checkpoint criterion changed")
    require(lock.get("run_count") == len(configs), "Validation lock has an incomplete trial count")
    require(lock.get("seed") == identity["seed"] and lock.get("fractions") == identity["fractions"]
            and lock.get("models") == identity["models"], "Validation lock condition identity changed")
    require(len(lock.get("runs", [])) == len(configs), "Validation lock run list is incomplete")
    for relative, expected_hash in lock["artifact_sha256"].items():
        require(legacy.file_sha256(study_dir / relative) == expected_hash,
                f"Validation artifact changed after lock: {relative}")
    expected_by_trial = {experiment.trial_dir(study_dir, config).name: config for config in configs}
    require({item["trial"] for item in lock["runs"]} == set(expected_by_trial), "Validation lock trials changed")
    for item in lock["runs"]:
        config = expected_by_trial[item["trial"]]
        require(item["configuration"] == config, "Validation lock configuration changed")
        record = experiment.read_trial(study_dir, config, verify_checkpoints=True)
        primary = record["evaluations"][PRIMARY_SELECTION]
        require(item["primary_checkpoint"] == {
            "selected_epoch": primary["selected_epoch"],
            "selected_validation_loss": primary["selected_validation_loss"],
            "validation_losses": primary["validation_losses"],
            "checkpoint_path": primary["checkpoint_path"],
            "checkpoint_sha256": primary["checkpoint_sha256"],
        }, "Validation primary checkpoint no longer matches its validation record")
    return lock


def test_marker(study_dir: Path, lock: dict[str, Any]) -> None:
    marker = {
        "schema": SCHEMA,
        "validation_selection_fingerprint": lock["fingerprint"],
        "started_at": legacy.utc_now(),
        "training_closed": True,
        "test_used_for_model_or_checkpoint_selection": False,
    }
    path = study_dir / "final_evaluation_started.json"
    if path.exists():
        saved = experiment.read_json(path)
        require(saved.get("schema") == SCHEMA and saved.get("validation_selection_fingerprint") == lock["fingerprint"]
                and saved.get("training_closed") is True
                and saved.get("test_used_for_model_or_checkpoint_selection") is False,
                "Test marker belongs to another validation lock")
    else:
        immutable_or_same(path, marker)


def validate_test_metrics(metrics: dict[str, Any], config: dict[str, Any]) -> None:
    optional_sign = set(legacy.GLOBAL_METRIC_NAMES[:4]) if config["observation"]["g05_count_per_sample"] == 0 else set()
    for name in (*legacy.METRIC_NAMES, "observed_sample_fraction", "observations_per_sample"):
        require(name in metrics, f"Missing final metric: {name}")
        value = metrics[name]
        if name in optional_sign:
            require(value is None, f"Unobserved G05 metric must be N/A: {name}")
        else:
            require(isinstance(value, (int, float)) and math.isfinite(value), f"Nonfinite final metric: {name}")
    count = config["observation"]["g05_count_per_sample"]
    require(metrics["observed_sample_fraction"] == float(count > 0), "Observed sample fraction changed")
    require(metrics["observations_per_sample"] == count, "Observation count changed")


def evaluation_identity(
    *,
    lock: dict[str, Any],
    config: dict[str, Any],
    trial: str,
    objective: str,
    validation: dict[str, Any],
    test_count: int,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "validation_selection_fingerprint": lock["fingerprint"],
        "split": "test",
        "sample_count": test_count,
        "trial": trial,
        "model": config["model"]["name"],
        "fraction": config["observation"]["g05_fraction"],
        "g05_count_per_sample": config["observation"]["g05_count_per_sample"],
        "seed": config["training"]["seed"],
        "checkpoint_selection": objective,
        "selected_epoch": validation["selected_epoch"],
        "selected_validation_loss": validation["selected_validation_loss"],
        "validation_losses": validation["validation_losses"],
        "checkpoint_path": validation["checkpoint_path"],
        "checkpoint_sha256": validation["checkpoint_sha256"],
    }


def test_evaluation(
    study_dir: Path,
    *,
    data_path: Path,
    split: physics.DataSplit,
    stats: physics.NormalizationStats,
    device: torch.device,
    lock: dict[str, Any],
    configs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Open the test split only after the immutable validation lock is verified."""
    test_marker(study_dir, lock)
    arrays = physics.load_dataset(data_path)
    records: list[dict[str, Any]] = []
    by_fraction = {fraction: [config for config in configs if config["observation"]["g05_fraction"] == fraction]
                   for fraction in FRACTIONS}
    for fraction in FRACTIONS:
        test = physics.prepare_dataset(arrays, split.test, stats, fraction)
        require(len(test) == len(split.test), "Test dataset count changed")
        for config in by_fraction[fraction]:
            trial = experiment.trial_dir(study_dir, config).name
            validation_record = experiment.read_trial(study_dir, config, verify_checkpoints=True)
            weights = physics.LossWeights(**config["training"]["loss_weights"])
            for objective in CHECKPOINT_SELECTIONS:
                validation = validation_record["evaluations"][objective]
                identity = evaluation_identity(
                    lock=lock, config=config, trial=trial, objective=objective,
                    validation=validation, test_count=len(test),
                )
                path = study_dir / "final" / "evaluations" / f"{trial}__{objective}.json"
                if path.exists():
                    record = experiment.read_json(path)
                    experiment.unseal(record)
                    require(all(record.get(key) == value for key, value in identity.items()),
                            f"Saved test evaluation identity differs: {path}")
                    validate_test_metrics(record["metrics"], config)
                    legacy.validate_loss_values(record["losses"], config, f"saved test losses {trial}/{objective}")
                else:
                    experiment.strict_seed(config["training"]["seed"])
                    model, loaded_stats, checkpoint = legacy.load_trained_model(study_dir / validation["checkpoint_path"], device)
                    require(loaded_stats.to_dict() == stats.to_dict(), "Checkpoint normalization differs from train-only stats")
                    require(checkpoint["selected_epoch"] == validation["selected_epoch"], "Checkpoint epoch changed")
                    require(checkpoint["selected_validation_loss"] == validation["selected_validation_loss"],
                            "Checkpoint validation loss changed")
                    losses = legacy.run_epoch(
                        model,
                        physics.create_data_loader(test, config["training"]["batch_size"], device=device),
                        weights=weights,
                    )
                    loss_values = asdict(losses)
                    legacy.validate_loss_values(loss_values, config, f"test losses {trial}/{objective}")
                    metrics = physics.evaluate_model(
                        model, test, loaded_stats, batch_size=config["training"]["batch_size"], weights=weights,
                    )
                    validate_test_metrics(metrics, config)
                    record = experiment.seal({
                        **identity,
                        "evaluated_at": legacy.utc_now(),
                        "losses": loss_values,
                        "metrics": metrics,
                        "test_used_for_model_or_checkpoint_selection": False,
                    })
                    immutable_or_same(path, record)
                    print(
                        f"TEST {trial} {objective} | structure={losses.structure:.6f} "
                        f"position_3d={metrics['mean_position_3d_error']:.6f}",
                        flush=True,
                    )
                    del model
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                records.append(record)
        del test
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return records


def flat_test_row(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "trial": record["trial"],
        "model": record["model"],
        "fraction": record["fraction"],
        "g05_count_per_sample": record["g05_count_per_sample"],
        "seed": record["seed"],
        "checkpoint_selection": record["checkpoint_selection"],
        "selected_epoch": record["selected_epoch"],
        "selected_validation_loss": record["selected_validation_loss"],
        "checkpoint_sha256": record["checkpoint_sha256"],
        **{f"validation_loss_{name}": value for name, value in record["validation_losses"].items()},
        **{f"test_loss_{name}": value for name, value in record["losses"].items()},
        **record["metrics"],
    }


def finalize_results(study_dir: Path, lock: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    expected_count = len(FRACTIONS) * len(MODELS) * len(CHECKPOINT_SELECTIONS)
    require(len(records) == expected_count, "Final test record count is incomplete")
    paths = sorted((study_dir / "final" / "evaluations").glob("*.json"))
    require(len(paths) == expected_count, "Final evaluation files are incomplete")
    artifacts = {path.relative_to(study_dir).as_posix(): legacy.file_sha256(path) for path in paths}
    final_path = study_dir / "final" / "test_result.json"
    if final_path.exists():
        final = experiment.read_json(final_path)
        experiment.unseal(final)
        require(final.get("schema") == SCHEMA and final.get("validation_selection_fingerprint") == lock["fingerprint"],
                "Saved final result belongs to another validation lock")
        require(final.get("artifact_sha256") == artifacts, "Final evaluation artifacts changed")
    else:
        final = experiment.seal({
            "schema": SCHEMA,
            "completed_at": legacy.utc_now(),
            "validation_selection_fingerprint": lock["fingerprint"],
            "test_used_for_model_or_checkpoint_selection": False,
            "primary_checkpoint_selection": PRIMARY_SELECTION,
            "record_count": len(records),
            "records": records,
            "artifact_sha256": artifacts,
        })
        immutable_or_same(final_path, final)
    rows = [flat_test_row(record) for record in records]
    legacy.atomic_write_csv(study_dir / "test_results.csv", rows)
    legacy.atomic_write_csv(study_dir / "test_primary_best_structure.csv",
                            [row for row in rows if row["checkpoint_selection"] == PRIMARY_SELECTION])
    return final


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    device = resolve_device(args.device)
    study_dir = args.study_dir.resolve()
    parent_dir = args.parent_study.resolve()
    data_path = args.data.resolve()
    ensure_isolated_output(study_dir, args.seed, parent_dir)
    parent, selection, candidate = load_parent(parent_dir)
    environment = configure_runtime(parent, device)
    source_audit = audit_sources(parent_dir, parent)
    arrays, split, stats, data_audit = verify_data(parent, data_path)
    configs = run_configurations(parent, candidate, args.seed)
    preflight = {
        "seed": args.seed,
        "fractions": list(FRACTIONS),
        "models": list(MODELS),
        "selected_hyperparameters": candidate,
        "parent_selection_fingerprint": selection["fingerprint"],
        "data": data_audit,
        "environment": environment,
        "source_audit": source_audit,
        "output": str(study_dir),
    }
    if args.check_only:
        print(legacy.canonical_json(preflight), flush=True)
        return
    with legacy.experiment_locks(study_dir):
        identity = initialize_output(
            study_dir,
            parent_dir=parent_dir,
            parent=parent,
            selection=selection,
            candidate=candidate,
            data_audit=data_audit,
            source_audit=source_audit,
            environment=environment,
            configs=configs,
            split=split,
            stats=stats,
            seed=args.seed,
        )
        lock_path = study_dir / "validation_selection.json"
        if lock_path.exists():
            lock = validate_validation_lock(study_dir, identity=identity, configs=configs)
            print("VALIDATION LOCK already present; no training will run", flush=True)
        else:
            require(not (study_dir / "final_evaluation_started.json").exists(),
                    "Test evaluation started before the validation lock was created")
            print(
                f"TRAINING START | seed={args.seed} candidate={candidate['id']} "
                f"fractions={list(FRACTIONS)} models={list(MODELS)} device={device} | test sealed",
                flush=True,
            )
            for fraction in FRACTIONS:
                data = development_data(arrays, split, stats, fraction)
                for config in (item for item in configs if item["observation"]["g05_fraction"] == fraction):
                    experiment.train_validation_run(study_dir, config, data, device)
                    experiment.refresh_trial_table(study_dir)
                del data
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            lock = make_validation_lock(study_dir, identity=identity, configs=configs)
            print(f"VALIDATION LOCKED | trials={lock['run_count']} | test remains unused during training", flush=True)
        # Do not keep the all-sample container around as test evaluation starts.
        del arrays
        if device.type == "cuda":
            torch.cuda.empty_cache()
        records = test_evaluation(
            study_dir,
            data_path=data_path,
            split=split,
            stats=stats,
            device=device,
            lock=lock,
            configs=configs,
        )
        final = finalize_results(study_dir, lock, records)
    print(
        f"COMPLETE | seed={args.seed} validation_trials={len(configs)} "
        f"test_records={final['record_count']} primary={PRIMARY_SELECTION} | {study_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
