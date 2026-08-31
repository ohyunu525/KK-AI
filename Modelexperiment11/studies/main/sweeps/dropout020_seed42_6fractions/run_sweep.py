"""Run the selected Experiment11 setting for seed 42 at six G05 fractions.

This is a validation-only training sweep.  It deliberately writes outside the
sealed parent study's runs directory and never constructs a test TensorDataset.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "Codes"))

import ModelExperiment10 as legacy  # noqa: E402
import ModelExperiment11 as experiment  # noqa: E402
import NewLearning9 as physics  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402


PARENT_STUDY = ROOT / "Modelexperiment11" / "studies" / "main"
SWEEP_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT / "Models" / "charge_dataset_5charges_v9.npz"
FRACTIONS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
SEED = 42
CANDIDATE_ID = "dropout020"
MODELS = tuple(legacy.DEFAULT_MODELS)
TEXT_SUFFIXES = {".csv", ".json", ".md", ".py", ".txt"}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def compatible_hash_mode(path: Path, expected: str) -> str | None:
    """Accept an exact hash or a Git-only LF/CRLF checkout conversion."""
    raw = path.read_bytes()
    if sha256_bytes(raw) == expected:
        return "exact"
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return None
    lf = raw.replace(b"\r\n", b"\n")
    if sha256_bytes(lf) == expected:
        return "crlf_to_lf"
    crlf = lf.replace(b"\n", b"\r\n")
    if sha256_bytes(crlf) == expected:
        return "lf_to_crlf"
    return None


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def compatible_trial(study: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Verify a parent-study trial while tolerating Git text line endings."""
    directory = experiment.trial_dir(PARENT_STUDY, config)
    record = experiment.read_json(directory / "result.json")
    experiment.unseal(record)
    require(record["schema"] == experiment.SCHEMA, "Parent trial schema mismatch")
    require(record["status"] == "validation_complete", "Parent trial is incomplete")
    require(record["configuration"] == config, "Parent trial configuration mismatch")
    require(record["test_evaluated"] is False, "A tuning trial evaluated test data")
    require(
        compatible_hash_mode(directory / "history.json", record["history_sha256"]) is not None,
        "Parent trial history hash mismatch",
    )
    for evaluation in record["evaluations"].values():
        checkpoint = PARENT_STUDY / evaluation["checkpoint_path"]
        if checkpoint.exists():
            require(
                compatible_hash_mode(checkpoint, evaluation["checkpoint_sha256"]) == "exact",
                "Parent checkpoint hash mismatch",
            )
    return record


def compatible_scores(study: dict[str, Any], candidate_ids: list[str], seeds: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate_id in candidate_ids:
        records = [
            compatible_trial(study, config)
            for config in experiment.expected_trials(study, [candidate_id], seeds)
        ]
        values = [record["evaluations"]["structure"]["selected_validation_loss"] for record in records]
        seed_scores = [
            float(
                np.mean(
                    [
                        record["evaluations"]["structure"]["selected_validation_loss"]
                        for record in records
                        if record["configuration"]["training"]["seed"] == seed
                    ]
                )
            )
            for seed in seeds
        ]
        rows.append(
            {
                "candidate_id": candidate_id,
                "score": float(np.mean(values)),
                "seed_std": legacy.sample_std(seed_scores),
                "seed_scores": dict(zip(map(str, seeds), seed_scores)),
                "trial_count": len(records),
                "seeds": seeds,
                "by_condition": {
                    f"{model}/g{fraction:g}": float(
                        np.mean(
                            [
                                record["evaluations"]["structure"]["selected_validation_loss"]
                                for record in records
                                if record["configuration"]["model"]["name"] == model
                                and record["configuration"]["observation"]["g05_fraction"] == fraction
                            ]
                        )
                    )
                    for model, fraction in itertools.product(
                        study["spec"]["models"], study["spec"]["fractions"]
                    )
                },
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            row["score"],
            row["candidate_id"] != "baseline",
            row["candidate_id"],
        ),
    )


def validate_parent_selection(study: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    promotion = experiment.read_json(PARENT_STUDY / "promotion.json")
    selection = experiment.read_json(PARENT_STUDY / "selection.json")
    experiment.unseal(promotion)
    experiment.unseal(selection)
    require(selection["study_fingerprint"] == study["fingerprint"], "Selection/study mismatch")
    require(selection["test_used_for_selection"] is False, "Selection used test data")
    require(
        compatible_hash_mode(PARENT_STUDY / "promotion.json", selection["promotion_sha256"])
        is not None,
        "Promotion hash mismatch beyond line endings",
    )
    artifact_modes: dict[str, str] = {}
    for relative, expected in selection["artifact_sha256"].items():
        path = PARENT_STUDY / relative
        if not path.exists() and path.suffix == ".pt":
            artifact_modes[relative] = "missing_parent_checkpoint_not_reused"
            continue
        mode = compatible_hash_mode(path, expected)
        require(mode is not None, f"Frozen parent artifact changed: {relative}")
        artifact_modes[relative] = mode
    selected, ranking, reason = experiment.choose_setting(
        compatible_scores(study, promotion["promoted"], selection["seeds"]),
        study["spec"],
    )
    require(selected == selection["selected_candidate_id"], "Selected candidate no longer reproduces")
    require(ranking == selection["ranking"], "Selection ranking no longer reproduces")
    require(reason == selection["reason"], "Selection reason no longer reproduces")
    return selection, {
        "promotion_hash_mode": compatible_hash_mode(
            PARENT_STUDY / "promotion.json", selection["promotion_sha256"]
        ),
        "artifact_count": len(artifact_modes),
        "exact_artifact_count": sum(mode == "exact" for mode in artifact_modes.values()),
        "newline_normalized_artifact_count": sum(
            mode in {"crlf_to_lf", "lf_to_crlf"} for mode in artifact_modes.values()
        ),
        "missing_parent_checkpoint_count": sum(
            mode == "missing_parent_checkpoint_not_reused" for mode in artifact_modes.values()
        ),
        "parent_checkpoint_weights_reused": False,
        "ranking_reproduced": True,
    }


def validate_sources(study: dict[str, Any]) -> dict[str, Any]:
    audit: dict[str, Any] = {}
    training_sources = {"ModelExperiment10.py", "ModelExperiment11.py", "NewLearning9.py"}
    for name, expected in study["source_sha256"].items():
        live = ROOT / "Codes" / name
        snapshot = PARENT_STUDY / "sources" / name
        live_mode = compatible_hash_mode(live, expected)
        snapshot_mode = compatible_hash_mode(snapshot, expected)
        require(live.read_bytes() == snapshot.read_bytes(), f"Live/snapshot source differs: {name}")
        if name in training_sources:
            require(live_mode is not None and snapshot_mode is not None, f"Training source changed: {name}")
        audit[name] = {
            "recorded_sha256": expected,
            "live_hash_mode": live_mode or "recorded_byte_hash_unrecoverable_after_checkout",
            "snapshot_hash_mode": snapshot_mode or "recorded_byte_hash_unrecoverable_after_checkout",
            "live_snapshot_bytes_equal": True,
            "used_during_training": name in training_sources,
        }
    return audit


def validate_data(study: dict[str, Any]) -> tuple[physics.DatasetArrays, dict[str, Any]]:
    require(DATA_PATH.is_file(), f"Dataset is missing: {DATA_PATH}")
    require(legacy.file_sha256(DATA_PATH) == study["data"]["sha256"], "Dataset file hash mismatch")
    arrays = physics.load_dataset(DATA_PATH)
    require(experiment.dataset_hashes(arrays) == study["data"]["array_hashes"], "Dataset array hash mismatch")
    split = physics.create_data_split(len(arrays.target), study["split_seed"])
    for phase in ("train", "validation", "test"):
        require(
            np.array_equal(getattr(split, phase), np.asarray(study["split"][phase])),
            f"Reproduced {phase} split differs",
        )
    stats = physics.calculate_normalization_stats(arrays, split.train)
    require(stats.to_dict() == study["normalization"], "Train-only normalization differs")
    require(
        experiment.read_json(PARENT_STUDY / "normalization.json") == study["normalization"],
        "Normalization artifact differs",
    )
    return arrays, {
        "relocated_path": str(DATA_PATH),
        "recorded_path": study["data"]["path"],
        "sha256": study["data"]["sha256"],
        "array_hashes": study["data"]["array_hashes"],
        "generation_seed": study["data"]["generation_seed"],
        "split_seed": study["split_seed"],
        "split_reproduced": True,
        "parent_split_archive_present": (PARENT_STUDY / "split_indices.npz").exists(),
        "normalization_reproduced": True,
    }


def development_data(
    study: dict[str, Any], arrays: physics.DatasetArrays, fraction: float
) -> experiment.DevelopmentData:
    stats = legacy.normalization_from_config(study)
    return experiment.DevelopmentData(
        physics.prepare_dataset(
            arrays,
            np.asarray(study["split"]["train"]),
            stats,
            fraction,
        ),
        physics.prepare_dataset(
            arrays,
            np.asarray(study["split"]["validation"]),
            stats,
            fraction,
        ),
    )


def initialize_manifest(identity: dict[str, Any]) -> dict[str, Any]:
    path = SWEEP_DIR / "sweep.json"
    if path.exists():
        manifest = experiment.read_json(path)
        experiment.unseal(manifest)
        require(manifest["identity"] == identity, "Existing sweep has a different identity")
        return manifest
    manifest = experiment.seal(
        {
            "schema": "model-experiment11-selected-fraction-sweep-v1",
            "created_at": legacy.utc_now(),
            "identity": identity,
        }
    )
    experiment.immutable_json(path, manifest)
    return manifest


def summarize_run(record: dict[str, Any]) -> dict[str, Any]:
    config = record["configuration"]
    return {
        "trial": experiment.trial_key(
            config["tuning"]["candidate_id"],
            config["model"]["name"],
            config["observation"]["g05_fraction"],
            config["training"]["seed"],
        ),
        "model": config["model"]["name"],
        "fraction": config["observation"]["g05_fraction"],
        "g05_count_per_sample": config["observation"]["g05_count_per_sample"],
        "seed": config["training"]["seed"],
        "training_result": record["training_result"],
        "evaluations": record["evaluations"],
        "test_evaluated": record["test_evaluated"],
        "result_fingerprint": record["fingerprint"],
    }


def finalize(
    study: dict[str, Any], manifest: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    summaries: list[dict[str, Any]] = []
    artifact_hashes: dict[str, str] = {}
    for fraction, model in itertools.product(FRACTIONS, MODELS):
        config = experiment.configuration(study, candidate, model, fraction, SEED)
        record = experiment.read_trial(SWEEP_DIR, config, verify_checkpoints=True)
        require(record["test_evaluated"] is False, "Sweep trial evaluated test data")
        summaries.append(summarize_run(record))
        directory = experiment.trial_dir(SWEEP_DIR, config)
        for path in sorted(item for item in directory.iterdir() if item.is_file()):
            artifact_hashes[path.relative_to(SWEEP_DIR).as_posix()] = legacy.file_sha256(path)
    rows = experiment.refresh_trial_table(SWEEP_DIR)
    require(len(summaries) == len(FRACTIONS) * len(MODELS), "Unexpected completed run count")
    require(len(rows) == len(summaries) * len(legacy.CHECKPOINT_SELECTIONS), "Unexpected trial-table row count")
    artifact_hashes["trials.csv"] = legacy.file_sha256(SWEEP_DIR / "trials.csv")
    artifact_hashes["run_sweep.py"] = legacy.file_sha256(Path(__file__).resolve())
    result_path = SWEEP_DIR / "sweep_result.json"
    if result_path.exists():
        result = experiment.read_json(result_path)
        experiment.unseal(result)
        require(result["manifest_fingerprint"] == manifest["fingerprint"], "Result/manifest mismatch")
        require(result["run_count"] == len(summaries), "Stored result run count mismatch")
        return result
    result = experiment.seal(
        {
            "schema": "model-experiment11-selected-fraction-sweep-result-v1",
            "completed_at": legacy.utc_now(),
            "manifest_fingerprint": manifest["fingerprint"],
            "candidate": candidate,
            "seed": SEED,
            "fractions": list(FRACTIONS),
            "models": list(MODELS),
            "run_count": len(summaries),
            "checkpoint_evaluation_count": len(rows),
            "test_evaluated": False,
            "runs": summaries,
            "artifact_sha256": artifact_hashes,
        }
    )
    experiment.immutable_json(result_path, result)
    return result


def main() -> None:
    torch.set_num_threads(10)
    torch.set_num_interop_threads(10)
    require(torch.cuda.is_available(), "The recorded Experiment11 CUDA device is unavailable")
    device = torch.device("cuda")
    experiment.strict_seed(physics.DATA_SPLIT_SEED)

    study = experiment.read_json(PARENT_STUDY / "study.json")
    experiment.unseal(study)
    require(study["schema"] == experiment.SCHEMA, "Unsupported parent study")
    experiment.validate_spec(study["spec"])
    require(study["candidates"] == experiment.make_candidates(study["spec"]), "Candidate table changed")
    actual_environment = experiment.environment(device)
    require(actual_environment == study["environment"], "Runtime/device differs from Experiment11")

    source_audit = validate_sources(study)
    arrays, data_audit = validate_data(study)
    selection, selection_audit = validate_parent_selection(study)
    require(selection["selected_candidate_id"] == CANDIDATE_ID, "Unexpected selected candidate")
    candidate = next(item for item in study["candidates"] if item["id"] == CANDIDATE_ID)
    require(candidate == selection["selected_hyperparameters"], "Selected hyperparameters changed")
    require(MODELS == tuple(study["spec"]["models"]), "Experiment11 model pair changed")

    identity = {
        "parent_study_path": str(PARENT_STUDY),
        "parent_study_fingerprint": study["fingerprint"],
        "selection_fingerprint": selection["fingerprint"],
        "selected_candidate": candidate,
        "seed": SEED,
        "fractions": list(FRACTIONS),
        "models": list(MODELS),
        "charge_count": physics.CHARGE_COUNT,
        "max_epochs": study["legacy_protocol"]["training"]["max_epochs"],
        "batch_size": study["legacy_protocol"]["training"]["batch_size"],
        "early_stopping_patience": study["legacy_protocol"]["training"]["regularization"][
            "early_stopping_patience"
        ],
        "environment": actual_environment,
        "data_audit": data_audit,
        "source_audit": source_audit,
        "selection_audit": selection_audit,
        "policy": {
            "training_data": "frozen train split only",
            "checkpoint_selection": "frozen validation split only",
            "test_tensor_dataset_constructed": False,
            "test_evaluated": False,
            "output_isolated_from_parent_runs": True,
        },
    }
    manifest = initialize_manifest(identity)

    print(
        f"SWEEP START | candidate={CANDIDATE_ID} seed={SEED} "
        f"fractions={list(FRACTIONS)} models={list(MODELS)} device={device}",
        flush=True,
    )
    with legacy.experiment_locks(SWEEP_DIR):
        for fraction in FRACTIONS:
            data = development_data(study, arrays, fraction)
            for model in MODELS:
                config = experiment.configuration(study, candidate, model, fraction, SEED)
                require(config["training"]["seed"] == SEED, "Seed changed in configuration")
                require(
                    math.isclose(config["observation"]["g05_fraction"], fraction, abs_tol=0.0),
                    "Fraction changed in configuration",
                )
                experiment.train_validation_run(SWEEP_DIR, config, data, device)
                experiment.refresh_trial_table(SWEEP_DIR)
            del data
            torch.cuda.empty_cache()
        result = finalize(study, manifest, candidate)
    print(
        f"SWEEP COMPLETE | runs={result['run_count']} "
        f"checkpoint_evaluations={result['checkpoint_evaluation_count']} test_evaluated=False",
        flush=True,
    )


if __name__ == "__main__":
    main()
