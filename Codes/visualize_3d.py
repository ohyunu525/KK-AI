from __future__ import annotations

"""Visualize a real test-set reconstruction from a trained checkpoint.

The script reuses the current training code instead of defining a second model
or preprocessing pipeline. It supports the checkpoint formats produced by
``Codes/ModelExperiment.py`` and complete/composed v3 checkpoints produced by
``Codes/NewLearning8.py``.

Targets are ordered lexicographically by (x, y, z), and current evaluation does
not perform a charge permutation. This file therefore keeps direct q1->q1 and
q2->q2 correspondence. Evaluation's global +/- sign alignment is retained for
numeric parity, but G05=0 figures display the prediction as a +/- equivalence
class so an oracle-aligned sign is never presented as an absolute prediction.
"""

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader


SCRIPT_DIR = Path(__file__).resolve().parent
if (SCRIPT_DIR / "NewLearning8.py").is_file():
    PROJECT_DIR = SCRIPT_DIR.parent
    CODES_DIR = SCRIPT_DIR
else:
    PROJECT_DIR = SCRIPT_DIR
    CODES_DIR = PROJECT_DIR / "Codes"
if str(CODES_DIR) not in sys.path:
    sys.path.insert(0, str(CODES_DIR))

try:
    import NewLearning8 as physics
    import ModelExperiment as experiment
except ModuleNotFoundError as error:
    raise ModuleNotFoundError(
        "visualize_3d.py must remain in the project root beside Codes/."
    ) from error


V3_ARCHITECTURE = "physics-separated G00 structure + G05 global-sign v3"
PAIRING_POLICY = (
    "identity pairing from lexicographic target slots "
    "(evaluation performs no permutation matching)"
)


@dataclass(frozen=True)
class RuntimeContext:
    checkpoint_path: Path
    checkpoint_family: str
    checkpoint_id: str
    checkpoint_epoch: int | None
    model_name: str
    model: torch.nn.Module
    arrays: physics.DatasetArrays
    stats: physics.NormalizationStats
    test_indices: np.ndarray
    split_seed: int | None
    seed: int | None
    g05_fraction: float
    g05_count: int
    g05_candidate_count: int
    align_global_charge_sign: Callable[[np.ndarray, np.ndarray], np.ndarray]
    notes: tuple[str, ...]
    figure_warnings: tuple[str, ...]


@dataclass(frozen=True)
class InferenceBatch:
    positions: np.ndarray
    raw_charges: np.ndarray
    displayed_charges: np.ndarray
    target_positions: np.ndarray
    target_charges: np.ndarray
    masks: np.ndarray
    global_sign_aligned: np.ndarray


@dataclass(frozen=True)
class SelectedSample:
    mode: str
    test_index: int
    dataset_index: int
    true_positions: np.ndarray
    predicted_positions: np.ndarray
    true_charges: np.ndarray
    raw_predicted_charges: np.ndarray
    predicted_charges: np.ndarray
    position_errors: np.ndarray
    mean_position_error: float
    error_rank: int
    error_percentile: float
    test_count: int
    test_mean_position_error: float
    test_median_position_error: float
    g05_mask: np.ndarray
    global_sign_aligned: bool


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not read JSON metadata: {path}") from error
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def extract_state_dict(checkpoint: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """Return a full state dict and reject purpose-specific components."""
    for key in ("model_state_dict", "state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping) and value and all(
            isinstance(tensor, torch.Tensor) for tensor in value.values()
        ):
            state_dict = dict(value)
            break
    else:
        if checkpoint and all(
            isinstance(tensor, torch.Tensor) for tensor in checkpoint.values()
        ):
            state_dict = dict(checkpoint)
        elif "component_state_dict" in checkpoint:
            raise ValueError(
                "The selected file is a purpose-specific component checkpoint. "
                "Choose a *_composed.pt, canonical v3 checkpoint, best.pt, or "
                "latest.pt containing a complete model_state_dict."
            )
        else:
            raise KeyError("Checkpoint does not contain a complete model state_dict")

    module_prefix = [name.startswith("module.") for name in state_dict]
    if all(module_prefix):
        return {
            name.removeprefix("module."): value for name, value in state_dict.items()
        }
    if any(module_prefix):
        raise ValueError("Checkpoint mixes DataParallel and non-DataParallel keys")
    return state_dict


def load_state_dict_strict(
    model: torch.nn.Module,
    state_dict: Mapping[str, torch.Tensor],
    checkpoint_path: Path,
) -> None:
    try:
        model.load_state_dict(dict(state_dict), strict=True)
    except RuntimeError as error:
        raise RuntimeError(
            "Checkpoint state_dict is incompatible with the model architecture "
            f"selected from project metadata: {checkpoint_path}"
        ) from error


def scalar_from_mapping(values: Mapping[str, Any], key: str, source: str) -> float:
    if key not in values:
        raise KeyError(f"Missing normalization value {key!r} in {source}")
    array = np.asarray(values[key])
    if array.size != 1:
        raise ValueError(f"Normalization value {key!r} is not scalar in {source}")
    result = float(array.reshape(()))
    if not np.isfinite(result):
        raise ValueError(f"Normalization value {key!r} is non-finite in {source}")
    return result


def vector_from_mapping(
    values: Mapping[str, Any], key: str, length: int, source: str
) -> np.ndarray:
    if key not in values:
        raise KeyError(f"Missing normalization value {key!r} in {source}")
    result = np.asarray(values[key], dtype=np.float32)
    if result.shape != (length,) or not np.isfinite(result).all():
        raise ValueError(
            f"Invalid normalization vector {key!r} in {source}: {result.shape}"
        )
    return result


def normalization_from_mapping(
    values: Mapping[str, Any], source: str
) -> physics.NormalizationStats:
    stats = physics.NormalizationStats(
        g00_mean=scalar_from_mapping(values, "g00_mean", source),
        g00_std=scalar_from_mapping(values, "g00_std", source),
        g05_value_scale=scalar_from_mapping(values, "g05_value_scale", source),
        position_mean=vector_from_mapping(values, "position_mean", 6, source),
        position_std=vector_from_mapping(values, "position_std", 6, source),
        charge_scale=scalar_from_mapping(values, "charge_scale", source),
    )
    if stats.g00_std <= 0 or stats.g05_value_scale <= 0 or stats.charge_scale <= 0:
        raise ValueError(f"Normalization scales must be positive in {source}")
    if np.any(stats.position_std <= 0):
        raise ValueError(f"Position standard deviations must be positive in {source}")
    return stats


def assert_normalization_matches(
    saved: physics.NormalizationStats,
    recalculated: physics.NormalizationStats,
    source: str,
) -> None:
    pairs = (
        ("g00_mean", saved.g00_mean, recalculated.g00_mean),
        ("g00_std", saved.g00_std, recalculated.g00_std),
        ("g05_value_scale", saved.g05_value_scale, recalculated.g05_value_scale),
        ("position_mean", saved.position_mean, recalculated.position_mean),
        ("position_std", saved.position_std, recalculated.position_std),
        ("charge_scale", saved.charge_scale, recalculated.charge_scale),
    )
    for name, expected, actual in pairs:
        if not np.allclose(expected, actual, rtol=2e-6, atol=2e-7):
            raise ValueError(
                f"Saved {name} does not match train-only normalization "
                f"recalculated from the selected dataset/split ({source})"
            )


def validate_split(
    train: np.ndarray,
    validation: np.ndarray,
    test: np.ndarray,
    sample_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    named = {"train": train, "validation": validation, "test": test}
    converted: dict[str, np.ndarray] = {}
    for name, indices in named.items():
        array = np.asarray(indices)
        if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
            raise ValueError(f"Saved {name} indices are not a 1D integer array")
        array = array.astype(np.int64, copy=False)
        if np.any((array < 0) | (array >= sample_count)):
            raise ValueError(f"Saved {name} indices are outside the dataset")
        converted[name] = array

    all_indices = np.concatenate(tuple(converted.values()))
    if all_indices.size != sample_count or np.unique(all_indices).size != sample_count:
        raise ValueError("Saved train/validation/test indices are not a full partition")
    return converted["train"], converted["validation"], converted["test"]


def indices_sha256(indices: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(indices).tobytes()).hexdigest()


def canonical_source_sha256(path: Path) -> str:
    """Hash source text after normalizing CRLF/CR to LF."""
    try:
        content = path.read_bytes()
    except OSError as error:
        raise RuntimeError(f"Could not hash source file: {path}") from error
    canonical = content.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(canonical).hexdigest()


def source_hash_matches(recorded_hash: str, path: Path) -> bool:
    """Accept either the exact bytes or the newline-canonical source bytes."""
    expected = recorded_hash.strip().lower()
    if not expected:
        return False
    current_hashes = {
        experiment.file_sha256(path).lower(),
        canonical_source_sha256(path).lower(),
    }
    return expected in current_hashes


def resolve_dataset_path(
    recorded_path: str | None, override_path: Path | None
) -> Path:
    if override_path is not None:
        path = override_path.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Dataset override does not exist: {path}")
        return path

    candidates: list[Path] = []
    if recorded_path:
        recorded = Path(recorded_path).expanduser()
        candidates.append(recorded)
        # The repository was relocated. The caller verifies the exact saved
        # SHA-256 before accepting this filename in the current Models folder.
        candidates.append(PROJECT_DIR / "Models" / recorded.name)
    candidates.append(Path(physics.DEFAULT_DATA_PATH))

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate.resolve()
    attempted = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Could not locate the dataset recorded by the checkpoint. Tried:\n  "
        f"{attempted}\nPass the exact file with --data."
    )


def model_experiment_research_issues(
    checkpoint_path: Path,
    checkpoint: Mapping[str, Any],
    config_path: Path,
    config: Mapping[str, Any],
    run_fingerprint: str,
) -> list[str]:
    """Return reasons a checkpoint is not the completed, reported best model."""
    issues: list[str] = []
    run_dir = config_path.parent
    run_id = run_dir.name
    if checkpoint_path.name.lower() != "best.pt":
        issues.append("the selected checkpoint is not best.pt")
    if checkpoint_path.parent.name != run_id:
        issues.append("the checkpoint directory does not match the saved run id")

    status_path = run_dir / "status.json"
    result_path = run_dir / "result.json"
    status: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    if status_path.is_file():
        try:
            status = load_json(status_path)
        except (RuntimeError, TypeError) as error:
            issues.append(f"status.json is unreadable ({error})")
    else:
        issues.append("status.json is missing")
    if result_path.is_file():
        try:
            result = load_json(result_path)
        except (RuntimeError, TypeError) as error:
            issues.append(f"result.json is unreadable ({error})")
    else:
        issues.append("result.json is missing")

    if status is not None:
        if str(status.get("run_id", "")) != run_id:
            raise ValueError("status.json belongs to a different run")
        if status.get("status") != "completed":
            issues.append(f"run status is {status.get('status')!r}, not 'completed'")
    if result is not None:
        if str(result.get("run_id", "")) != run_id:
            raise ValueError("result.json belongs to a different run")
        if result.get("run_fingerprint") != run_fingerprint:
            raise ValueError("result.json fingerprint differs from the checkpoint")
        result_config = result.get("configuration")
        if not isinstance(result_config, Mapping):
            raise TypeError("result.json is missing its run configuration")
        if experiment.object_fingerprint(result_config) != run_fingerprint:
            raise ValueError("result.json run configuration fingerprint is invalid")
        if dict(result_config) != dict(config):
            raise ValueError("result.json and config.json run configurations differ")
        if result.get("status") != "completed":
            issues.append("result.json is not marked completed")

    checkpoint_epoch = checkpoint.get("epoch")
    if checkpoint_epoch is None:
        issues.append("the checkpoint does not record its epoch")
    else:
        checkpoint_epoch = int(checkpoint_epoch)

    status_best_epoch: int | None = None
    if status is not None:
        if status.get("best_epoch") is None:
            issues.append("status.json does not record best_epoch")
        else:
            status_best_epoch = int(status["best_epoch"])

    result_best_epoch: int | None = None
    if result is not None:
        training_result = result.get("training_result")
        if not isinstance(training_result, Mapping) or training_result.get(
            "best_epoch"
        ) is None:
            issues.append("result.json does not record training_result.best_epoch")
        else:
            result_best_epoch = int(training_result["best_epoch"])
        artifacts = result.get("artifacts")
        if isinstance(artifacts, Mapping) and artifacts.get("best_checkpoint"):
            recorded_best = Path(str(artifacts["best_checkpoint"]))
            if recorded_best.name.lower() != "best.pt" or recorded_best.parent.name != run_id:
                raise ValueError("result.json points to a best checkpoint from another run")
        else:
            issues.append("result.json does not identify the reported best checkpoint")

    if (
        status_best_epoch is not None
        and result_best_epoch is not None
        and status_best_epoch != result_best_epoch
    ):
        raise ValueError("status.json and result.json disagree on best_epoch")
    expected_best_epoch = (
        result_best_epoch if result_best_epoch is not None else status_best_epoch
    )
    if (
        checkpoint_epoch is not None
        and expected_best_epoch is not None
        and checkpoint_epoch != expected_best_epoch
    ):
        issues.append(
            f"checkpoint epoch {checkpoint_epoch} is not reported best_epoch "
            f"{expected_best_epoch}"
        )
    return issues


def assert_dataset_metadata(
    arrays: physics.DatasetArrays, metadata: Mapping[str, Any]
) -> None:
    expected_shapes = {
        "g00_shape": tuple(arrays.g00.shape),
        "g05_shape": tuple(arrays.g05.shape),
        "target_shape": tuple(arrays.target.shape),
    }
    for key, actual in expected_shapes.items():
        if key in metadata and tuple(metadata[key]) != actual:
            raise ValueError(f"Dataset {key} does not match the saved protocol")
    if "target_fields" in metadata and tuple(metadata["target_fields"]) != tuple(
        physics.TARGET_FIELDS
    ):
        raise ValueError("Saved target field order is incompatible with the project")
    if "g05_fields" in metadata and tuple(metadata["g05_fields"]) != tuple(
        physics.G05_FIELDS
    ):
        raise ValueError("Saved G05 field order is incompatible with the project")


def find_model_experiment_config(
    run_fingerprint: str, checkpoint_path: Path
) -> Path:
    results_root = PROJECT_DIR / "Results" / "model_experiments"
    candidates: list[Path] = []
    if checkpoint_path.parent.parent != checkpoint_path.parent:
        candidates.append(
            results_root
            / checkpoint_path.parent.parent.name
            / "runs"
            / checkpoint_path.parent.name
            / "config.json"
        )
    if results_root.exists():
        candidates.extend(results_root.glob("*/runs/*/config.json"))

    matches: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen or not candidate.exists():
            continue
        seen.add(candidate)
        try:
            config = load_json(candidate)
        except RuntimeError:
            continue
        if experiment.object_fingerprint(config) == run_fingerprint:
            matches.append(candidate)

    if not matches:
        raise FileNotFoundError(
            "This ModelExperiment checkpoint stores only a run fingerprint and "
            "weights. Its matching Results/model_experiments/.../config.json "
            "could not be found, so routing, fraction, split, and normalization "
            "cannot be reconstructed without guessing."
        )
    preferred = [path for path in matches if path.parent.name == checkpoint_path.parent.name]
    if len(preferred) == 1:
        return preferred[0]
    if len(matches) != 1:
        raise RuntimeError(
            "Multiple experiment configs match this checkpoint fingerprint: "
            + ", ".join(str(path) for path in matches)
        )
    return matches[0]


def load_model_experiment_context(
    checkpoint_path: Path,
    checkpoint: Mapping[str, Any],
    state_dict: Mapping[str, torch.Tensor],
    data_override: Path | None,
    device: torch.device,
    exploratory: bool = False,
) -> RuntimeContext:
    run_fingerprint = str(checkpoint["run_fingerprint"])
    config_path = find_model_experiment_config(run_fingerprint, checkpoint_path)
    config = load_json(config_path)
    if experiment.object_fingerprint(config) != run_fingerprint:
        raise RuntimeError("Run config fingerprint changed while it was being read")

    experiment_results_dir = config_path.parents[2]
    protocol_path = experiment_results_dir / "protocol.json"
    normalization_path = experiment_results_dir / "normalization.json"
    split_path = experiment_results_dir / "split_indices.npz"
    protocol_document = load_json(protocol_path)
    protocol = protocol_document.get("protocol")
    if not isinstance(protocol, Mapping):
        raise TypeError(f"Missing protocol object: {protocol_path}")
    protocol_fingerprint = str(config.get("protocol_fingerprint", ""))
    if protocol_document.get("protocol_fingerprint") != protocol_fingerprint:
        raise ValueError("Run config and protocol.json fingerprints differ")
    if experiment.object_fingerprint(protocol) != protocol_fingerprint:
        raise ValueError("Saved protocol content does not match its fingerprint")

    model_metadata = config.get("model")
    observation_metadata = config.get("observation")
    training_metadata = config.get("training")
    if not all(
        isinstance(value, Mapping)
        for value in (model_metadata, observation_metadata, training_metadata)
    ):
        raise TypeError("ModelExperiment config is missing model/observation/training")
    model_name = str(model_metadata["name"])
    if model_name not in experiment.MODEL_REGISTRY:
        raise KeyError(
            f"Checkpoint model {model_name!r} is not registered in ModelExperiment.py"
        )
    model_spec = experiment.MODEL_REGISTRY[model_name]
    verification_issues = model_experiment_research_issues(
        checkpoint_path, checkpoint, config_path, config, run_fingerprint
    )
    saved_input_policy = model_metadata.get("input_policy")
    if not isinstance(saved_input_policy, Mapping):
        verification_issues.append("the run config does not record model.input_policy")
    elif dict(saved_input_policy) != dict(model_spec.input_policy):
        verification_issues.append(
            "the current model registry input policy differs from the saved policy"
        )

    dataset_metadata = protocol.get("dataset")
    split_metadata = protocol.get("data_split")
    code_metadata = protocol.get("code")
    if not isinstance(dataset_metadata, Mapping) or not isinstance(
        split_metadata, Mapping
    ):
        raise TypeError("Protocol is missing dataset or data_split metadata")

    dataset_path = resolve_dataset_path(
        str(dataset_metadata.get("path", "")) or None, data_override
    )
    expected_data_hash = str(dataset_metadata.get("sha256", ""))
    if not expected_data_hash:
        raise ValueError("ModelExperiment protocol does not contain a dataset SHA-256")
    actual_data_hash = experiment.file_sha256(dataset_path)
    if actual_data_hash.lower() != expected_data_hash.lower():
        raise ValueError(
            f"Dataset SHA-256 mismatch for {dataset_path}; refusing to infer on "
            "a dataset different from training/evaluation"
        )
    arrays = physics.load_dataset(dataset_path)
    assert_dataset_metadata(arrays, dataset_metadata)

    try:
        with np.load(split_path, allow_pickle=False) as archive:
            train, validation, test = validate_split(
                archive["train"],
                archive["validation"],
                archive["test"],
                arrays.g00.shape[0],
            )
    except (OSError, KeyError, ValueError) as error:
        raise RuntimeError(f"Could not load the exact saved split: {split_path}") from error
    for name, indices in (("train", train), ("validation", validation), ("test", test)):
        expected_hash = str(split_metadata.get(f"{name}_indices_sha256", ""))
        if not expected_hash or indices_sha256(indices) != expected_hash:
            raise ValueError(f"Saved {name} split does not match protocol.json")

    saved_stats = normalization_from_mapping(
        load_json(normalization_path), str(normalization_path)
    )
    protocol_normalization = protocol.get("normalization")
    if isinstance(protocol_normalization, Mapping):
        protocol_stats = normalization_from_mapping(
            protocol_normalization, str(protocol_path)
        )
        assert_normalization_matches(saved_stats, protocol_stats, str(protocol_path))
    recalculated_stats = physics.calculate_normalization_stats(arrays, train)
    assert_normalization_matches(saved_stats, recalculated_stats, str(split_path))

    fraction = float(observation_metadata["g05_fraction"])
    candidate_count = int(observation_metadata["candidate_count"])
    g05_count = int(observation_metadata["g05_count_per_sample"])
    if candidate_count != arrays.g05.shape[1]:
        raise ValueError("Checkpoint G05 candidate count does not match the dataset")
    expected_g05_count = physics.g05_count_for_fraction(fraction, candidate_count)
    if g05_count != expected_g05_count:
        raise ValueError("Checkpoint G05 count is inconsistent with its fraction")

    notes: list[str] = []
    source_paths = {
        "model_experiment_sha256": Path(experiment.__file__).resolve(),
        "physics_protocol_sha256": Path(physics.__file__).resolve(),
    }
    if not isinstance(code_metadata, Mapping):
        verification_issues.append("the protocol does not record source hashes")
    else:
        config_code_hash = str(config.get("code_sha256", ""))
        protocol_code_hash = str(code_metadata.get("model_experiment_sha256", ""))
        if config_code_hash and protocol_code_hash:
            if config_code_hash.lower() != protocol_code_hash.lower():
                raise ValueError(
                    "Run config and protocol disagree on the training source hash"
                )
        elif not config_code_hash:
            verification_issues.append("the run config does not record code_sha256")

        for key, source_path in source_paths.items():
            saved_hash = str(code_metadata.get(key, ""))
            if not saved_hash:
                verification_issues.append(f"the protocol does not record {key}")
            elif not source_hash_matches(saved_hash, source_path):
                verification_issues.append(
                    f"current {source_path.name} semantics are unverified because "
                    f"{key} differs"
                )

    figure_warnings: tuple[str, ...] = ()
    if verification_issues:
        issue_text = "; ".join(dict.fromkeys(verification_issues))
        if not exploratory:
            raise RuntimeError(
                "Research checkpoint verification failed: "
                f"{issue_text}. Use --exploratory only for a clearly labeled, "
                "non-reporting visualization."
            )
        notes.append(f"Exploratory checkpoint accepted: {issue_text}.")
        figure_warnings = (
            "EXPLORATORY / UNVERIFIED MODEL",
            "Not the verified completed best checkpoint used for reported test metrics",
        )

    model = model_spec.factory().to(device)
    load_state_dict_strict(model, state_dict, checkpoint_path)
    model.eval()

    return RuntimeContext(
        checkpoint_path=checkpoint_path,
        checkpoint_family="ModelExperiment v1",
        checkpoint_id=run_fingerprint[:12],
        checkpoint_epoch=(
            int(checkpoint["epoch"]) if checkpoint.get("epoch") is not None else None
        ),
        model_name=model_name,
        model=model,
        arrays=arrays,
        stats=saved_stats,
        test_indices=test,
        split_seed=int(split_metadata["seed"]),
        seed=int(training_metadata["seed"]),
        g05_fraction=fraction,
        g05_count=g05_count,
        g05_candidate_count=candidate_count,
        align_global_charge_sign=experiment.align_global_charge_sign,
        notes=tuple(dict.fromkeys(notes)),
        figure_warnings=figure_warnings,
    )


def load_v3_context(
    checkpoint_path: Path,
    checkpoint: Mapping[str, Any],
    state_dict: Mapping[str, torch.Tensor],
    data_override: Path | None,
    device: torch.device,
    allow_unverifiable_v3: bool = False,
) -> RuntimeContext:
    if checkpoint.get("model_architecture") != V3_ARCHITECTURE:
        raise ValueError("Checkpoint is not a recognized NewLearning8 v3 full model")

    dataset_metadata = checkpoint.get("dataset")
    if not isinstance(dataset_metadata, Mapping):
        dataset_metadata = {}
    recorded_data_path = checkpoint.get("dataset_path", dataset_metadata.get("path"))
    dataset_path = resolve_dataset_path(
        None if recorded_data_path is None else str(recorded_data_path), data_override
    )
    expected_data_hash = str(
        checkpoint.get("dataset_sha256", dataset_metadata.get("sha256", ""))
    ).strip()
    if expected_data_hash:
        actual_data_hash = experiment.file_sha256(dataset_path)
        if actual_data_hash.lower() != expected_data_hash.lower():
            raise ValueError(
                f"Dataset SHA-256 mismatch for {dataset_path}; refusing to infer on "
                "a dataset different from the v3 checkpoint"
            )
    arrays = physics.load_dataset(dataset_path)
    expected_grid_shape = checkpoint.get("grid_shape")
    if expected_grid_shape is not None and tuple(expected_grid_shape) != tuple(
        arrays.g00.shape[1:]
    ):
        raise ValueError("Checkpoint grid shape does not match the dataset")
    expected_candidate_count = int(checkpoint["g05_candidate_count"])
    if expected_candidate_count != arrays.g05.shape[1]:
        raise ValueError("Checkpoint G05 candidate count does not match the dataset")
    if tuple(checkpoint.get("target_fields", ())) != tuple(physics.TARGET_FIELDS):
        raise ValueError("Checkpoint target order is incompatible with the project")
    if tuple(checkpoint.get("g05_fields", ())) != tuple(physics.G05_FIELDS):
        raise ValueError("Checkpoint G05 field order is incompatible with the project")

    split_metadata = checkpoint.get("data_split")
    if not isinstance(split_metadata, Mapping):
        split_metadata = {}
    split_seed_value = checkpoint.get(
        "data_split_seed", split_metadata.get("seed")
    )
    split_seed = None if split_seed_value is None else int(split_seed_value)

    split_indices = checkpoint.get("split_indices")
    if not isinstance(split_indices, Mapping):
        split_indices = {}
    exact_values: dict[str, Any] = {}
    for name in ("train", "validation", "test"):
        if name in split_indices:
            exact_values[name] = split_indices[name]
        elif f"{name}_indices" in split_metadata:
            exact_values[name] = split_metadata[f"{name}_indices"]
        elif f"{name}_indices" in checkpoint:
            exact_values[name] = checkpoint[f"{name}_indices"]
    if exact_values and len(exact_values) != 3:
        raise ValueError("v3 checkpoint stores only part of the exact data split")

    if len(exact_values) == 3:
        train, validation, test = validate_split(
            exact_values["train"],
            exact_values["validation"],
            exact_values["test"],
            arrays.g00.shape[0],
        )
        has_exact_split = True
    else:
        if split_seed is None:
            split_seed = int(physics.DATA_SPLIT_SEED)
        split = physics.create_data_split(arrays.g00.shape[0], split_seed)
        train, validation, test = validate_split(
            split.train, split.validation, split.test, arrays.g00.shape[0]
        )
        has_exact_split = False

    expected_split_hashes: dict[str, str] = {}
    for name in ("train", "validation", "test"):
        key = f"{name}_indices_sha256"
        value = split_metadata.get(key, checkpoint.get(key, ""))
        if value:
            expected_split_hashes[name] = str(value).lower()
    if expected_split_hashes and len(expected_split_hashes) != 3:
        raise ValueError("v3 checkpoint stores only part of the split hash metadata")
    if len(expected_split_hashes) == 3:
        for name, indices in (
            ("train", train),
            ("validation", validation),
            ("test", test),
        ):
            if indices_sha256(indices) != expected_split_hashes[name]:
                raise ValueError(
                    f"v3 {name} split does not match its checkpoint SHA-256"
                )
    has_verified_split = has_exact_split or (
        split_seed_value is not None and len(expected_split_hashes) == 3
    )

    missing_provenance: list[str] = []
    if not expected_data_hash:
        missing_provenance.append("dataset SHA-256")
    if not has_verified_split:
        missing_provenance.append("exact split indices or seed plus all split hashes")
    notes: list[str] = []
    figure_warnings: tuple[str, ...] = ()
    if missing_provenance:
        missing_text = ", ".join(missing_provenance)
        if not allow_unverifiable_v3:
            raise RuntimeError(
                "Legacy v3 checkpoint provenance is incomplete "
                f"({missing_text}). Refusing research/PPT output by default. "
                "Use --allow-unverifiable-v3 only when the warning watermark is "
                "acceptable."
            )
        notes.append(
            "Legacy v3 provenance override: missing "
            f"{missing_text}; the selected dataset and split seed were accepted only "
            "after train-only normalization matched."
        )
        figure_warnings = (
            "UNVERIFIED LEGACY V3 PROVENANCE",
            "Dataset identity and/or exact held-out split were not recorded",
        )

    saved_stats = normalization_from_mapping(checkpoint, str(checkpoint_path))
    recalculated_stats = physics.calculate_normalization_stats(arrays, train)
    assert_normalization_matches(saved_stats, recalculated_stats, str(checkpoint_path))

    fraction = float(checkpoint["g05_fraction"])
    g05_count = int(checkpoint["g05_count"])
    expected_count = physics.g05_count_for_fraction(fraction, arrays.g05.shape[1])
    if g05_count != expected_count:
        raise ValueError("Checkpoint G05 count is inconsistent with its fraction")

    model = physics.ChargeNet().to(device)
    load_state_dict_strict(model, state_dict, checkpoint_path)
    model.eval()
    return RuntimeContext(
        checkpoint_path=checkpoint_path,
        checkpoint_family="NewLearning8 physics-separated v3",
        checkpoint_id=re.sub(
            r"[^A-Za-z0-9]+", "-", checkpoint_path.stem
        ).strip("-")[-24:],
        checkpoint_epoch=None,
        model_name="physics_separated_v3",
        model=model,
        arrays=arrays,
        stats=saved_stats,
        test_indices=test,
        split_seed=split_seed,
        seed=int(checkpoint["seed"]) if checkpoint.get("seed") is not None else None,
        g05_fraction=fraction,
        g05_count=g05_count,
        g05_candidate_count=arrays.g05.shape[1],
        align_global_charge_sign=physics.align_global_charge_sign,
        notes=tuple(notes),
        figure_warnings=figure_warnings,
    )


def restore_runtime_context(
    checkpoint_path: Path,
    data_override: Path | None,
    device: torch.device,
    exploratory: bool = False,
    allow_unverifiable_v3: bool = False,
) -> RuntimeContext:
    checkpoint_path = checkpoint_path.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    checkpoint = experiment.load_torch_checkpoint(checkpoint_path, device)
    state_dict = extract_state_dict(checkpoint)

    if "run_fingerprint" in checkpoint:
        return load_model_experiment_context(
            checkpoint_path,
            checkpoint,
            state_dict,
            data_override,
            device,
            exploratory,
        )
    if checkpoint.get("model_architecture") == V3_ARCHITECTURE:
        return load_v3_context(
            checkpoint_path,
            checkpoint,
            state_dict,
            data_override,
            device,
            allow_unverifiable_v3,
        )
    raise ValueError(
        "Unsupported checkpoint format. This script accepts current "
        "verified ModelExperiment best checkpoints and complete/composed "
        "NewLearning8 v3 checkpoints. Legacy v1/v2 checkpoints use different "
        "architectures and preprocessing and are intentionally not guessed. "
        "Non-reporting ModelExperiment checkpoints require --exploratory."
    )


def infer_indices(
    context: RuntimeContext,
    dataset_indices: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> InferenceBatch:
    dataset = physics.prepare_dataset(
        context.arrays, dataset_indices, context.stats, context.g05_fraction
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    context.model.to(device)
    context.model.eval()
    positions: list[np.ndarray] = []
    charges: list[np.ndarray] = []
    target_positions: list[np.ndarray] = []
    target_charges: list[np.ndarray] = []
    masks: list[np.ndarray] = []

    # Keep no_grad explicit as required by the visualization contract.
    with torch.no_grad():
        for g00, g05, g05_mask, position_target, charge_target in loader:
            output = context.model(
                g00.to(device, non_blocking=True),
                g05.to(device, non_blocking=True),
                g05_mask.to(device, non_blocking=True),
            )
            reconstructed_charge = physics.reconstruct_charges(
                output.magnitude,
                output.relative_sign_logit,
                output.global_sign_logit,
            )
            positions.append(output.position.cpu().numpy())
            charges.append(reconstructed_charge.cpu().numpy())
            target_positions.append(position_target.numpy())
            target_charges.append(charge_target.numpy())
            masks.append(g05_mask.numpy())

    position_prediction = (
        np.concatenate(positions) * context.stats.position_std
        + context.stats.position_mean
    )
    raw_charge_prediction = np.concatenate(charges) * context.stats.charge_scale
    position_target = (
        np.concatenate(target_positions) * context.stats.position_std
        + context.stats.position_mean
    )
    charge_target = np.concatenate(target_charges) * context.stats.charge_scale
    g05_mask = np.concatenate(masks)

    finite_values = (
        position_prediction,
        raw_charge_prediction,
        position_target,
        charge_target,
    )
    if not all(np.isfinite(value).all() for value in finite_values):
        raise FloatingPointError("Model inference produced non-finite values")

    expected_target = context.arrays.target[dataset_indices]
    expected_position = expected_target[:, physics.POSITION_INDICES]
    expected_charge = expected_target[:, physics.CHARGE_INDICES]
    if not np.allclose(position_target, expected_position, rtol=2e-6, atol=2e-6):
        raise RuntimeError("Position denormalization does not reproduce raw targets")
    if not np.allclose(charge_target, expected_charge, rtol=2e-6, atol=2e-6):
        raise RuntimeError("Charge denormalization does not reproduce raw targets")

    observed = g05_mask.sum(axis=(1, 2)) > 0
    displayed_charges = raw_charge_prediction.copy()
    global_sign_aligned = ~observed
    if np.any(global_sign_aligned):
        displayed_charges[global_sign_aligned] = context.align_global_charge_sign(
            raw_charge_prediction[global_sign_aligned],
            charge_target[global_sign_aligned],
        )
    return InferenceBatch(
        positions=position_prediction,
        raw_charges=raw_charge_prediction,
        displayed_charges=displayed_charges,
        target_positions=position_target,
        target_charges=charge_target,
        masks=g05_mask,
        global_sign_aligned=global_sign_aligned,
    )


def sample_errors(batch: InferenceBatch) -> tuple[np.ndarray, np.ndarray]:
    predicted = batch.positions.reshape(-1, 2, 3)
    target = batch.target_positions.reshape(-1, 2, 3)
    per_charge = np.linalg.norm(predicted - target, axis=2)
    return per_charge, per_charge.mean(axis=1)


def select_sample(
    context: RuntimeContext,
    sample_index: int | None,
    sample_mode: str,
    device: torch.device,
    batch_size: int,
) -> SelectedSample:
    test_count = len(context.test_indices)
    if test_count == 0:
        raise ValueError("The saved test split is empty")
    dataset_indices = context.test_indices
    batch = infer_indices(context, dataset_indices, device, batch_size)
    per_charge_errors, mean_errors = sample_errors(batch)

    if sample_index is not None:
        if not 0 <= sample_index < test_count:
            raise IndexError(
                f"--sample-index is test-local and must be in [0, {test_count - 1}]"
            )
        selected_batch_index = sample_index
        selected_test_index = sample_index
        mode = "index"
    else:
        if sample_mode == "best":
            selected_batch_index = int(np.argmin(mean_errors))
        elif sample_mode == "worst":
            selected_batch_index = int(np.argmax(mean_errors))
        elif sample_mode == "median":
            median_error = float(np.median(mean_errors))
            selected_batch_index = int(np.argmin(np.abs(mean_errors - median_error)))
        else:
            raise ValueError(f"Unknown sample mode: {sample_mode}")
        selected_test_index = selected_batch_index
        mode = sample_mode

    index = selected_batch_index
    selected_error = float(mean_errors[index])
    error_rank = int(np.count_nonzero(mean_errors < selected_error)) + 1
    error_percentile = 100.0 * float(
        np.count_nonzero(mean_errors <= selected_error)
    ) / test_count
    return SelectedSample(
        mode=mode,
        test_index=selected_test_index,
        dataset_index=int(dataset_indices[index]),
        true_positions=batch.target_positions[index].reshape(2, 3).copy(),
        predicted_positions=batch.positions[index].reshape(2, 3).copy(),
        true_charges=batch.target_charges[index].copy(),
        raw_predicted_charges=batch.raw_charges[index].copy(),
        predicted_charges=batch.displayed_charges[index].copy(),
        position_errors=per_charge_errors[index].copy(),
        mean_position_error=selected_error,
        error_rank=error_rank,
        error_percentile=error_percentile,
        test_count=test_count,
        test_mean_position_error=float(np.mean(mean_errors)),
        test_median_position_error=float(np.median(mean_errors)),
        g05_mask=batch.masks[index, :, 0].astype(bool, copy=True),
        global_sign_aligned=bool(batch.global_sign_aligned[index]),
    )


def format_position(position: np.ndarray) -> str:
    return f"({position[0]: .6f}, {position[1]: .6f}, {position[2]: .6f})"


def print_console_summary(
    context: RuntimeContext, sample: SelectedSample, device: torch.device
) -> None:
    for note in context.notes:
        print(f"WARNING: {note}")
    print("=" * 60)
    print("3D Reconstruction Visualization")
    print("=" * 60)
    print(f"Checkpoint: {context.checkpoint_path}")
    print(f"Checkpoint family: {context.checkpoint_family}")
    print(f"Model: {context.model_name}")
    print(f"Device: {device}")
    print(f"Sample index: {sample.test_index} (test-local)")
    print(f"Dataset index: {sample.dataset_index}")
    print(f"Sample mode: {sample.mode}")
    print(f"Provenance: synthetic held-out test sample ({sample.test_count} total)")
    print(
        "Selection criterion: mean of q1/q2 direct-pair 3D position errors; "
        f"rank={sample.error_rank}/{sample.test_count}, "
        f"empirical percentile={sample.error_percentile:.1f}%"
    )
    print(
        f"G05 fraction: {context.g05_fraction:.3f} "
        f"({context.g05_count}/{context.g05_candidate_count} points)"
    )
    print(f"Seed/checkpoint: {context.seed if context.seed is not None else 'N/A'}")
    print(f"Charge pairing: {PAIRING_POLICY}")

    for charge_index in range(2):
        print("")
        print(f"Charge {charge_index + 1}")
        print(f"True position : {format_position(sample.true_positions[charge_index])}")
        print(f"Pred position : {format_position(sample.predicted_positions[charge_index])}")
        print(f"True q        : {sample.true_charges[charge_index]: .6f}")
        if sample.global_sign_aligned:
            print(
                f"Raw model q   : {sample.raw_predicted_charges[charge_index]: .6f} "
                "(one representative of the +/- equivalence class)"
            )
            print(
                f"Eval-only q   : {sample.predicted_charges[charge_index]: .6f} "
                "(ground-truth oracle-aligned global sign)"
            )
        else:
            print(f"Pred q        : {sample.predicted_charges[charge_index]: .6f}")
        print(f"3D error      : {sample.position_errors[charge_index]: .6f}")

    print("")
    print(f"Mean 3D position error: {sample.mean_position_error: .6f}")
    print(
        f"Full test mean/median: {sample.test_mean_position_error: .6f} / "
        f"{sample.test_median_position_error: .6f}"
    )
    print("=" * 60)


def padded_limits(
    base_minimum: float,
    base_maximum: float,
    selected_values: np.ndarray,
    padding_fraction: float = 0.04,
) -> tuple[float, float]:
    minimum = min(base_minimum, float(np.min(selected_values)))
    maximum = max(base_maximum, float(np.max(selected_values)))
    span = maximum - minimum
    if span <= 0:
        span = 1.0
    padding = span * padding_fraction
    return minimum - padding, maximum + padding


def plot_limits(
    context: RuntimeContext, sample: SelectedSample
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    selected = np.vstack((sample.true_positions, sample.predicted_positions))
    x_limits = padded_limits(
        float(np.min(context.arrays.grid_x)),
        float(np.max(context.arrays.grid_x)),
        selected[:, 0],
    )
    y_limits = padded_limits(
        float(np.min(context.arrays.grid_y)),
        float(np.max(context.arrays.grid_y)),
        selected[:, 1],
    )
    dataset_z = context.arrays.target[:, (2, 6)]
    z_limits = padded_limits(
        min(0.0, float(np.min(dataset_z))),
        float(np.max(dataset_z)),
        selected[:, 2],
    )
    return x_limits, y_limits, z_limits


def charge_color(charge: float) -> str:
    if charge > 0:
        return "#c62828"
    if charge < 0:
        return "#1565c0"
    return "#616161"


def marker_size(_charge: float) -> float:
    """Use a fixed size; charge magnitude is reported numerically, not by area."""
    return 130.0


def build_output_path(
    output_dir: Path, context: RuntimeContext, sample: SelectedSample
) -> Path:
    def token(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9_-]+", "-", value).strip("-") or "unknown"

    seed = "seedNA" if context.seed is None else f"seed{context.seed}"
    fraction = f"g05_{int(round(context.g05_fraction * 100)):03d}"
    filename = (
        f"reconstruction_{token(sample.mode)}_{token(context.model_name)}_"
        f"{seed}_{fraction}_test{sample.test_index:04d}_"
        f"data{sample.dataset_index:05d}_ckpt{token(context.checkpoint_id)}.png"
    )
    return output_dir.expanduser().resolve() / filename


def render_plot(
    context: RuntimeContext,
    sample: SelectedSample,
    output_path: Path,
    show_g00: bool,
    show_g05: bool,
    dpi: int,
    show_window: bool,
) -> None:
    try:
        import matplotlib.cm as cm
        import matplotlib.colors as colors
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "matplotlib is required in the same Python environment as PyTorch."
        ) from error

    figure = plt.figure(figsize=(14.2, 8.8), facecolor="white")
    axis = figure.add_subplot(111, projection="3d")
    banner_line_count = len(context.figure_warnings) + int(
        sample.global_sign_aligned
    )
    axes_top = max(0.76, 0.90 - 0.04 * banner_line_count)
    figure.subplots_adjust(left=0.02, right=0.69, bottom=0.08, top=axes_top)

    grid_x, grid_y = np.meshgrid(context.arrays.grid_x, context.arrays.grid_y)
    plane_z = np.zeros_like(grid_x)
    if show_g00:
        g00 = context.arrays.g00[sample.dataset_index]
        # A dataset-wide scale keeps color intensity comparable across figures.
        minimum = float(np.min(context.arrays.g00))
        maximum = float(np.max(context.arrays.g00))
        if maximum <= minimum:
            maximum = minimum + 1.0
        normalization = colors.Normalize(vmin=minimum, vmax=maximum)
        colormap = plt.get_cmap("viridis")
        axis.plot_surface(
            grid_x,
            grid_y,
            plane_z,
            facecolors=colormap(normalization(g00)),
            rstride=1,
            cstride=1,
            linewidth=0,
            antialiased=False,
            shade=False,
            alpha=0.52,
        )
        scalar_mappable = cm.ScalarMappable(norm=normalization, cmap=colormap)
        scalar_mappable.set_array(g00)
        colorbar = figure.colorbar(
            scalar_mappable,
            ax=axis,
            shrink=0.52,
            pad=0.025,
            fraction=0.035,
        )
        colorbar.set_label("Observed G00 (dataset-wide scale)", fontsize=11)
        colorbar.ax.tick_params(labelsize=9)
    else:
        axis.plot_surface(
            grid_x,
            grid_y,
            plane_z,
            color="#9e9e9e",
            linewidth=0,
            shade=False,
            alpha=0.18,
        )

    g05_plotted = False
    if show_g05 and np.any(sample.g05_mask):
        raw_g05 = context.arrays.g05[sample.dataset_index]
        observed = raw_g05[sample.g05_mask]
        x_indices = np.rint(observed[:, 0]).astype(np.int64)
        y_indices = np.rint(observed[:, 1]).astype(np.int64)
        axis.scatter(
            context.arrays.grid_x[x_indices],
            context.arrays.grid_y[y_indices],
            np.zeros(len(observed)),
            marker="s",
            s=24,
            c="#212121",
            edgecolors="white",
            linewidths=0.55,
            depthshade=False,
        )
        g05_plotted = True

    x_limits, y_limits, z_limits = plot_limits(context, sample)
    annotation_offset = 0.025 * (z_limits[1] - z_limits[0])
    prediction_annotation_offset = 0.065 * (z_limits[1] - z_limits[0])
    horizontal_offset = 0.012 * (x_limits[1] - x_limits[0])
    line_styles = ("-", "--")
    for index in range(2):
        true_position = sample.true_positions[index]
        predicted_position = sample.predicted_positions[index]
        true_charge = float(sample.true_charges[index])
        raw_predicted_charge = float(sample.raw_predicted_charges[index])
        if sample.global_sign_aligned:
            plotted_predicted_charge = raw_predicted_charge
            predicted_color = "#757575"
            predicted_label = (
                f"Pred |q{index + 1}|={abs(raw_predicted_charge):.3f} (± pair)"
            )
        else:
            plotted_predicted_charge = float(sample.predicted_charges[index])
            predicted_color = charge_color(plotted_predicted_charge)
            predicted_label = f"Pred q{index + 1}={plotted_predicted_charge:+.3f}"
        axis.plot(
            (true_position[0], predicted_position[0]),
            (true_position[1], predicted_position[1]),
            (true_position[2], predicted_position[2]),
            linestyle=line_styles[index],
            color="#424242",
            linewidth=1.8,
            alpha=0.78,
        )
        axis.scatter(
            *true_position,
            marker="o",
            s=marker_size(true_charge),
            c=charge_color(true_charge),
            edgecolors="black",
            linewidths=1.0,
            depthshade=False,
        )
        axis.scatter(
            *predicted_position,
            marker="^",
            s=marker_size(plotted_predicted_charge),
            c=predicted_color,
            edgecolors="black",
            linewidths=1.0,
            depthshade=False,
        )
        axis.text(
            true_position[0] + horizontal_offset,
            true_position[1],
            true_position[2] + annotation_offset,
            f"True q{index + 1}={true_charge:+.3f}",
            fontsize=10.5,
            horizontalalignment="left",
        )
        axis.text(
            predicted_position[0] - horizontal_offset,
            predicted_position[1],
            predicted_position[2] + prediction_annotation_offset,
            predicted_label,
            fontsize=10.5,
            horizontalalignment="right",
        )

    axis.text2D(
        0.02,
        0.025,
        "Observation plane: z = 0",
        transform=axis.transAxes,
        fontsize=10,
        color="#424242",
    )
    axis.set_xlim(*x_limits)
    axis.set_ylim(*y_limits)
    axis.set_zlim(*z_limits)
    axis.set_box_aspect(
        (
            x_limits[1] - x_limits[0],
            y_limits[1] - y_limits[0],
            z_limits[1] - z_limits[0],
        )
    )
    axis.set_xlabel("x", fontsize=13, labelpad=10)
    axis.set_ylabel("y", fontsize=13, labelpad=10)
    # Axes3D can place the native z label behind an adjacent colorbar. A 2D
    # axes-relative label remains visible in both the plain-plane and G00 views.
    axis.set_zlabel("")
    axis.text2D(
        0.975,
        0.53,
        "z",
        transform=axis.transAxes,
        fontsize=13,
        rotation=90,
        ha="center",
        va="center",
    )
    axis.tick_params(labelsize=10)
    axis.set_title("3D Point-Charge Reconstruction", fontsize=16, pad=16)
    axis.view_init(elev=24, azim=-55)
    for pane in (axis.xaxis.pane, axis.yaxis.pane, axis.zaxis.pane):
        pane.set_alpha(0.045)

    predicted_legend_label = (
        "Predicted position / ± charge class"
        if sample.global_sign_aligned
        else "Predicted position"
    )
    predicted_legend_color = "#757575" if sample.global_sign_aligned else "white"
    legend_handles = [
        Line2D(
            [0], [0], marker="o", color="none", markerfacecolor="white",
            markeredgecolor="black", markersize=9, label="True position"
        ),
        Line2D(
            [0], [0], marker="^", color="none",
            markerfacecolor=predicted_legend_color,
            markeredgecolor="black", markersize=9, label=predicted_legend_label
        ),
        Line2D(
            [0], [0], marker="o", color="none", markerfacecolor=charge_color(1.0),
            markeredgecolor="black", markersize=8, label="Positive charge (+)"
        ),
        Line2D(
            [0], [0], marker="o", color="none", markerfacecolor=charge_color(-1.0),
            markeredgecolor="black", markersize=8, label="Negative charge (-)"
        ),
        Line2D(
            [0], [0], color="#424242", linewidth=1.8, label="Position error"
        ),
    ]
    if g05_plotted:
        legend_handles.append(
            Line2D(
                [0], [0], marker="s", color="none", markerfacecolor="#212121",
                markeredgecolor="white", markersize=7, label="Observed G05 point"
            )
        )
    axis.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.0, 0.98),
        fontsize=9.5,
        framealpha=0.92,
    )

    seed_text = "N/A" if context.seed is None else str(context.seed)
    epoch_text = (
        "N/A" if context.checkpoint_epoch is None else str(context.checkpoint_epoch)
    )
    info_lines = [
        "Synthetic held-out test sample",
        f"Selection: {sample.mode} among {sample.test_count} test samples",
        "Criterion: mean(q1/q2 direct-pair 3D errors)",
        f"Error rank/percentile: {sample.error_rank}/{sample.test_count} "
        f"({sample.error_percentile:.1f}%)",
        f"Full test mean/median: {sample.test_mean_position_error:.5f} / "
        f"{sample.test_median_position_error:.5f}",
        "",
        f"Sample index: {sample.test_index} (dataset {sample.dataset_index})",
        f"G05 fraction: {context.g05_fraction:.3f} "
        f"({context.g05_count}/{context.g05_candidate_count})",
        f"Seed: {seed_text}    Epoch: {epoch_text}",
        f"Checkpoint: {context.checkpoint_path.name}",
        f"Model: {context.model_name}",
        "",
        f"True q1: {sample.true_charges[0]:+.4f}",
        f"True q2: {sample.true_charges[1]:+.4f}",
    ]
    if sample.global_sign_aligned:
        info_lines.extend(
            (
                "Predicted charge class:",
                f"  ±({sample.raw_predicted_charges[0]:+.4f}, "
                f"{sample.raw_predicted_charges[1]:+.4f})",
                "Absolute/global sign: unidentifiable (N/A)",
            )
        )
    else:
        info_lines.extend(
            (
                f"Pred q1: {sample.predicted_charges[0]:+.4f}",
                f"Pred q2: {sample.predicted_charges[1]:+.4f}",
            )
        )
    info_lines.extend(
        (
            "",
            f"Charge 1 position error: {sample.position_errors[0]:.5f}",
            f"Charge 2 position error: {sample.position_errors[1]:.5f}",
            f"Mean 3D position error: {sample.mean_position_error:.5f}",
        )
    )
    figure.text(
        0.75,
        0.50,
        "\n".join(info_lines),
        ha="left",
        va="center",
        fontsize=10.1,
        linespacing=1.22,
        family="sans-serif",
        bbox={
            "boxstyle": "round,pad=0.65",
            "facecolor": "#fafafa",
            "edgecolor": "#bdbdbd",
            "alpha": 0.97,
        },
    )

    banner_lines = list(context.figure_warnings)
    if sample.global_sign_aligned:
        banner_lines.append(
            "G05=0: ABSOLUTE GLOBAL SIGN UNIDENTIFIABLE — "
            "gray predictions are a ± equivalence class"
        )
    if banner_lines:
        figure.text(
            0.5,
            0.965,
            "\n".join(banner_lines),
            ha="center",
            va="top",
            fontsize=11.5,
            weight="bold",
            color="#8d1b1b",
            bbox={
                "boxstyle": "round,pad=0.45",
                "facecolor": "#fff3e0",
                "edgecolor": "#d84315",
                "alpha": 0.98,
            },
        )
    if context.figure_warnings:
        figure.text(
            0.36,
            0.45,
            context.figure_warnings[0],
            ha="center",
            va="center",
            fontsize=23,
            weight="bold",
            color="#b71c1c",
            alpha=0.14,
            rotation=24,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    if show_window:
        plt.show()
    plt.close(figure)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
    return torch.device(name)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare true and inferred 3D charges for one real test sample."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--sample-index", type=int, help="Zero-based index within the saved test split"
    )
    selection.add_argument(
        "--sample-mode",
        choices=("best", "median", "worst"),
        default=None,
        help="Select by mean per-sample 3D position error over the full test set",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Exact dataset override; saved SHA/statistics are still verified",
    )
    parser.add_argument("--show-g00", action="store_true")
    parser.add_argument("--show-g05", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "Results" / "visualizations",
    )
    parser.add_argument("--dpi", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--exploratory",
        action="store_true",
        help=(
            "Allow a non-best, incomplete, or source-drifted ModelExperiment "
            "checkpoint and add an unverified watermark"
        ),
    )
    parser.add_argument(
        "--allow-unverifiable-v3",
        action="store_true",
        help=(
            "Allow a legacy v3 checkpoint without dataset/split provenance and "
            "add a warning watermark"
        ),
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save the PNG without opening an interactive window",
    )
    args = parser.parse_args(argv)
    if args.sample_index is not None and args.sample_index < 0:
        parser.error("--sample-index must be non-negative")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if not 72 <= args.dpi <= 600:
        parser.error("--dpi must be between 72 and 600")
    if args.sample_mode is None:
        args.sample_mode = "median"
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.no_show:
        try:
            import matplotlib
        except ModuleNotFoundError as error:
            raise ModuleNotFoundError(
                "matplotlib is required for visualization."
            ) from error
        matplotlib.use("Agg", force=True)

    device = resolve_device(args.device)
    context = restore_runtime_context(
        args.checkpoint,
        args.data,
        device,
        exploratory=args.exploratory,
        allow_unverifiable_v3=args.allow_unverifiable_v3,
    )
    sample = select_sample(
        context, args.sample_index, args.sample_mode, device, args.batch_size
    )
    print_console_summary(context, sample, device)
    output_path = build_output_path(args.output_dir, context, sample)
    render_plot(
        context,
        sample,
        output_path,
        show_g00=args.show_g00,
        show_g05=args.show_g05,
        dpi=args.dpi,
        show_window=not args.no_show,
    )
    print(f"Saved PNG: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
