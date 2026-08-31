"""Reproducible, validation-only tuning around the unchanged ModelExperiment10.

python Codes/ModelExperiment11.py run
python Codes/ModelExperiment11.py tune       # stops after locking the selection
python Codes/ModelExperiment11.py finalize   # requires a locked selection

All new artifacts default to Modelexperiment11/studies/main. No legacy file is
modified. Checkpoints retain the v10 inference format; training never invokes
v10's train_and_evaluate_run, which would evaluate test after every candidate.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
import re
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# This import sets CUBLAS_WORKSPACE_CONFIG before any CUDA operations.
import ModelExperiment10 as legacy
import NewLearning9 as physics
import generate_charge_dataset as generator
import numpy as np
import torch
from torch.utils.data import TensorDataset

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STUDY = ROOT / "Modelexperiment11" / "studies" / "main"
DEFAULT_CONFIG = ROOT / "Modelexperiment11" / "search_config.json"
SCHEMA = "model-experiment11-validation-tuning-v1"
SOURCES = ("ModelExperiment11.py", "ModelExperiment10.py", "NewLearning9.py", "generate_charge_dataset.py")
load_trained_model = legacy.load_trained_model


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(value: Any) -> str:
    return legacy.object_fingerprint(value)


def seal(value: dict) -> dict:
    if "fingerprint" in value:
        raise ValueError("Cannot seal an already sealed object")
    return {**value, "fingerprint": digest(value)}


def unseal(value: dict) -> dict:
    payload = {key: item for key, item in value.items() if key != "fingerprint"}
    if value.get("fingerprint") != digest(payload):
        raise RuntimeError("Artifact fingerprint mismatch")
    return payload


def immutable_json(path: Path, value: Any) -> None:
    if path.exists():
        if legacy.canonical_json(read_json(path)) != legacy.canonical_json(value):
            raise RuntimeError(f"Refusing to overwrite a different artifact: {path}")
    else:
        legacy.atomic_write_json(path, value)


def array_hash(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    return hashlib.sha256(str(value.dtype).encode() + str(value.shape).encode() + value.tobytes()).hexdigest()


def dataset_hashes(arrays: physics.DatasetArrays) -> dict:
    return {name: array_hash(getattr(arrays, name)) for name in ("g00", "g05", "target", "grid_x", "grid_y")}


def strict_seed(seed: int) -> None:
    legacy.set_reproducibility(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)


def environment(device: torch.device) -> dict:
    result = legacy.runtime_environment(device)
    result["deterministic_algorithms"] = "enabled, warn_only=False"
    result["cudnn_benchmark"] = torch.backends.cudnn.benchmark
    result["cudnn_deterministic"] = torch.backends.cudnn.deterministic
    return result


def validate_spec(spec: dict) -> None:
    fields = {"models", "fractions", "screen_seeds", "confirmation_seeds", "max_epochs", "batch_size",
              "early_stopping_patience", "top_k", "min_improvement_pct", "max_model_regression_pct",
              "search_seed", "anchors", "random_candidates", "search_space", "fresh_test"}
    if not isinstance(spec, dict) or set(spec) != fields:
        raise ValueError(f"Search configuration requires exactly: {sorted(fields)}")
    for name in ("max_epochs", "batch_size", "top_k"):
        if type(spec[name]) is not int or spec[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    for name in ("random_candidates", "early_stopping_patience"):
        if type(spec[name]) is not int or spec[name] < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    for name in ("min_improvement_pct", "max_model_regression_pct"):
        if type(spec[name]) not in (int, float) or not math.isfinite(spec[name]) or not 0 <= spec[name] < 100:
            raise ValueError(f"{name} must be finite and in [0,100)")
    if spec["models"] != list(legacy.DEFAULT_MODELS):
        raise ValueError("The common-setting comparison must include both original models in their fixed order")
    fractions = spec["fractions"]
    if (not isinstance(fractions, list) or not fractions or len(set(fractions)) != len(fractions)
            or any(type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= 1 for x in fractions)):
        raise ValueError("fractions must be a nonempty list of unique values in [0,1]")
    for key in ("screen_seeds", "confirmation_seeds"):
        if not isinstance(spec[key], list) or not spec[key]:
            raise ValueError(f"{key} must be nonempty")
    seeds = spec["screen_seeds"] + spec["confirmation_seeds"]
    if len(seeds) != len(set(seeds)) or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds):
        raise ValueError("Screening and confirmation seeds must be unique, disjoint uint32 integers")
    if type(spec["search_seed"]) is not int or not 0 <= spec["search_seed"] < 2**32:
        raise ValueError("search_seed must be a uint32 integer")
    space = spec["search_space"]
    if set(space) != {"learning_rate", "weight_decay", "structure_dropout"}:
        raise ValueError("Only learning_rate, weight_decay, structure_dropout may be searched")
    for key in ("learning_rate", "weight_decay"):
        bounds = space[key]
        if (not isinstance(bounds, list) or len(bounds) != 2
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in bounds)
                or not 0 < bounds[0] <= bounds[1]):
            raise ValueError(f"{key} needs positive finite log-uniform bounds")
    values = space["structure_dropout"]
    if (not isinstance(values, list) or not values or len(set(values)) != len(values)
            or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v < 1 for v in values)):
        raise ValueError("structure_dropout needs distinct probabilities in [0,1)")
    if not isinstance(spec["anchors"], list) or len(spec["anchors"]) < 2:
        raise ValueError("At least baseline and one comparison anchor are required")
    seen_ids, seen_values = set(), set()
    for candidate in spec["anchors"]:
        if set(candidate) != {"id", *space}:
            raise ValueError("Candidate must contain id and exactly the three hyperparameters")
        name = candidate["id"]
        if not isinstance(name, str) or re.fullmatch(r"[a-z][a-z0-9_-]{0,35}", name) is None or name in seen_ids:
            raise ValueError("Candidate ids must be unique safe short identifiers")
        seen_ids.add(name)
        for key in ("learning_rate", "weight_decay"):
            v = candidate[key]
            if type(v) not in (int, float) or not math.isfinite(v) or not space[key][0] <= v <= space[key][1]:
                raise ValueError(f"Candidate {name}: {key} is outside declared search bounds")
        if candidate["structure_dropout"] not in values:
            raise ValueError("Anchor dropout must be in the declared space")
        params = digest({key: candidate[key] for key in space})
        if params in seen_values:
            raise ValueError("Duplicate hyperparameter combinations")
        seen_values.add(params)
    baseline = {"id": "baseline", "learning_rate": 1e-3, "weight_decay": 1e-4, "structure_dropout": 0.0}
    if spec["anchors"][0] != baseline:
        raise ValueError("The first anchor must be the unchanged v10 baseline")
    if spec["top_k"] > len(spec["anchors"]) + spec["random_candidates"] - 1:
        raise ValueError("top_k exceeds the number of non-baseline candidates")
    fresh = spec["fresh_test"]
    if (set(fresh) != {"samples", "seed"} or type(fresh["samples"]) is not int or fresh["samples"] < 10
            or type(fresh["seed"]) is not int or not 0 <= fresh["seed"] < 2**32):
        raise ValueError("fresh_test requires >=10 samples and a uint32 seed")


def make_candidates(spec: dict) -> list[dict]:
    validate_spec(spec)
    result = copy.deepcopy(spec["anchors"])
    rng = np.random.default_rng(spec["search_seed"])
    for index in range(spec["random_candidates"]):
        name = f"random_{index + 1:02d}"
        if any(row["id"] == name for row in result):
            raise ValueError("Anchor id conflicts with a generated random candidate")
        candidate = {"id": name}
        for key in ("learning_rate", "weight_decay"):
            lo, hi = spec["search_space"][key]
            candidate[key] = float(f"{np.exp(rng.uniform(np.log(lo), np.log(hi))):.10g}")
        candidate["structure_dropout"] = float(rng.choice(spec["search_space"]["structure_dropout"]))
        if any(all(candidate[k] == row[k] for k in spec["search_space"]) for row in result):
            raise ValueError("Generated duplicate candidate; choose another search seed in a new study")
        result.append(candidate)
    return result


def validate_split(arrays: physics.DatasetArrays, split: physics.DataSplit) -> dict:
    all_indices = np.concatenate([getattr(split, phase) for phase in ("train", "validation", "test")])
    if (len(all_indices) != len(arrays.target) or len(np.unique(all_indices)) != len(all_indices)
            or not np.array_equal(np.sort(all_indices), np.arange(len(arrays.target)))):
        raise ValueError("Split must be disjoint and cover every sample exactly once")
    # Identical G00 also catches whole-charge sign inversions with the same field.
    hashes = [hashlib.sha256(row.tobytes()).hexdigest() for row in arrays.g00]
    groups = [{hashes[i] for i in getattr(split, phase)} for phase in ("train", "validation", "test")]
    if any(a & b for a, b in itertools.combinations(groups, 2)):
        raise ValueError("Identical physical inputs cross splits; use a grouped split in a separate study")
    return {"counts": {phase: len(getattr(split, phase)) for phase in ("train", "validation", "test")},
            "unique_g00_samples": len(set(hashes)), "cross_split_identical_g00": 0}


def initialize(study_dir: Path, spec: dict, data_path: Path, device: torch.device) -> dict:
    validate_spec(spec)
    strict_seed(physics.DATA_SPLIT_SEED)
    protocol_path = study_dir / "study.json"
    if protocol_path.exists():
        study = load_study(study_dir, device=device)
        if study["spec"] != spec or Path(study["data"]["path"]).resolve() != data_path.resolve():
            raise RuntimeError("Existing study has different input arguments; use a new directory")
        return study
    if study_dir.exists() and any(study_dir.iterdir()):
        raise RuntimeError("Nonempty study directory without study.json; refusing to mix artifacts")
    arrays = physics.load_dataset(data_path)
    split = physics.create_data_split(len(arrays.target))
    split_audit = validate_split(arrays, split)
    stats = physics.calculate_normalization_stats(arrays, split.train)
    with np.load(data_path, allow_pickle=False) as archive:
        generation_seed = int(archive["generation_seed"]) if "generation_seed" in archive else None
    if generation_seed == spec["fresh_test"]["seed"]:
        raise ValueError("Fresh holdout must have a different generation seed")
    # The simulator's fixed distribution/grid is required for the declared fresh IID holdout.
    axis = np.linspace(generator.GRID_MIN, generator.GRID_MAX, generator.GRID_SIZE)
    if (not np.array_equal(arrays.grid_x, axis) or not np.array_equal(arrays.grid_y, axis)
            or arrays.epsilon_0 != generator.EPSILON_0):
        raise ValueError("Fresh-test generation currently supports the original simulator grid/epsilon only")
    controls = legacy.RegularizationSettings(early_stopping_patience=spec["early_stopping_patience"])
    base = legacy.build_protocol(data_path=data_path, arrays=arrays, split=split, stats=stats,
                                 settings=physics.TrainingSettings(max_epochs=spec["max_epochs"],
                                                                   batch_size=spec["batch_size"]),
                                 regularization=controls, device=device)
    base["environment"] = environment(device)
    candidates = make_candidates(spec)
    study = seal({
        "schema": SCHEMA, "created_at": legacy.utc_now(), "spec": spec, "candidates": candidates,
        "data": {"path": str(data_path.resolve()), "sha256": legacy.file_sha256(data_path),
                 "array_hashes": dataset_hashes(arrays), "generation_seed": generation_seed},
        "split": {phase: getattr(split, phase).tolist() for phase in ("train", "validation", "test")},
        "split_audit": split_audit, "split_seed": physics.DATA_SPLIT_SEED,
        "normalization": stats.to_dict(), "legacy_protocol": base,
        "source_sha256": {name: legacy.file_sha256(ROOT / "Codes" / name) for name in SOURCES},
        "environment": environment(device),
        "selection_policy": {
            "metric": "validation.structure at best_structure.pt",
            "aggregation": "arithmetic mean, equal weight per model/fraction/seed",
            "common_hyperparameters": True, "checkpoint_tie": "first epoch",
            "candidate_tie": "baseline first, then candidate id",
            "promotion": "top_k non-baseline screening scores; baseline always confirmed",
            "final_ranking": "all screening + confirmation seeds; paired baseline in identical conditions",
            "guardrail": "per-model/fraction mean structure regression <= max_model_regression_pct",
            "fallback": "baseline unless eligible best beats baseline by min_improvement_pct",
            "secondary_only": "best_total and all physical/sign metrics; never switch criteria after test",
        },
        "test_policy": {
            "during_tuning": "no test TensorDataset, DataLoader, inference, or scoring",
            "final_gate": "immutable selection plus hashes of all selected and baseline checkpoints",
            "historical_holdout": "previously examined in older work; final comparison, not pristine evidence",
            "fresh_holdout": "generate after selection; fixed independent seed and original sensor coordinates",
            "no_refit": "evaluate frozen validation-selected whole-epoch checkpoints; do not train on validation",
        },
    })
    # Snapshot bytes and indices before publishing the study authority.
    for name in SOURCES:
        destination = study_dir / "sources" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / "Codes" / name).read_bytes())
    legacy.atomic_save_npz(study_dir / "split_indices.npz", **{p: getattr(split, p) for p in ("train", "validation", "test")})
    immutable_json(study_dir / "normalization.json", stats.to_dict())
    immutable_json(study_dir / "resolved_config.json", spec)
    immutable_json(protocol_path, study)
    legacy.atomic_write_csv(study_dir / "candidates.csv", candidates)
    print(f"INITIALIZED {study_dir} | {len(candidates)} candidates | test sealed", flush=True)
    return study


def load_study(study_dir: Path, device: torch.device | None = None) -> dict:
    study = read_json(study_dir / "study.json")
    unseal(study)
    if study["schema"] != SCHEMA:
        raise RuntimeError("Unsupported study schema")
    validate_spec(study["spec"])
    if study["candidates"] != make_candidates(study["spec"]):
        raise RuntimeError("Stored candidate table differs from its declared search")
    for name, expected in study["source_sha256"].items():
        if legacy.file_sha256(ROOT / "Codes" / name) != expected:
            raise RuntimeError(f"Source changed since initialization: {name}; use the snapshot or a new study")
        if legacy.file_sha256(study_dir / "sources" / name) != expected:
            raise RuntimeError(f"Source snapshot was modified: {name}")
    if legacy.file_sha256(Path(study["data"]["path"])) != study["data"]["sha256"]:
        raise RuntimeError("Dataset changed since initialization")
    if read_json(study_dir / "normalization.json") != study["normalization"]:
        raise RuntimeError("Stored train-only normalization changed")
    with np.load(study_dir / "split_indices.npz", allow_pickle=False) as archive:
        if any(not np.array_equal(archive[p], study["split"][p]) for p in ("train", "validation", "test")):
            raise RuntimeError("Stored split changed")
    if device is not None:
        strict_seed(physics.DATA_SPLIT_SEED)
        if environment(device) != study["environment"]:
            raise RuntimeError("Runtime/device changed; exact resume requires the recorded environment")
    return study


@dataclass(frozen=True)
class DevelopmentData:
    """Only train/validation tensors can cross the tuning training API."""
    train: TensorDataset
    validation: TensorDataset


def development_data(study: dict) -> dict[float, DevelopmentData]:
    arrays = physics.load_dataset(Path(study["data"]["path"]))
    stats = legacy.normalization_from_config(study)
    return {fraction: DevelopmentData(
        physics.prepare_dataset(arrays, np.asarray(study["split"]["train"]), stats, fraction),
        physics.prepare_dataset(arrays, np.asarray(study["split"]["validation"]), stats, fraction))
        for fraction in study["spec"]["fractions"]}


def configuration(study: dict, candidate: dict, model: str, fraction: float, seed: int) -> dict:
    base = copy.deepcopy(study["legacy_protocol"])
    base["training"]["learning_rate"] = candidate["learning_rate"]
    base["training"]["weight_decay"] = candidate["weight_decay"]
    base["training"]["regularization"]["structure_dropout"] = candidate["structure_dropout"]
    config = legacy.run_configuration(base, model_name=model, fraction=fraction, seed=seed)
    config["tuning"] = {"schema": SCHEMA, "study_fingerprint": study["fingerprint"],
                        "candidate_id": candidate["id"], "phase": "train_validation_only"}
    return config


def trial_key(candidate: str, model: str, fraction: float, seed: int) -> str:
    return f"{candidate}__{model}__g{fraction:g}__s{seed}"


def trial_dir(study_dir: Path, config: dict) -> Path:
    return study_dir / "runs" / trial_key(config["tuning"]["candidate_id"], config["model"]["name"],
                                        config["observation"]["g05_fraction"], config["training"]["seed"])


def expected_trials(study: dict, candidates: list[str], seeds: list[int]):
    by_id = {candidate["id"]: candidate for candidate in study["candidates"]}
    for candidate_id, seed, fraction, model in itertools.product(
            candidates, seeds, study["spec"]["fractions"], study["spec"]["models"]):
        yield configuration(study, by_id[candidate_id], model, fraction, seed)


def validate_development(config: dict, data: DevelopmentData) -> None:
    if not isinstance(data, DevelopmentData):
        raise TypeError("Training accepts DevelopmentData, never a train/validation/test tuple")
    count = config["observation"]["g05_count_per_sample"]
    candidates = config["observation"]["candidate_count"]
    for phase in ("train", "validation"):
        dataset = getattr(data, phase)
        if len(dataset) != config["split_counts"][phase]:
            raise ValueError(f"{phase} size does not match the common split")
        mask = dataset.tensors[2]
        if (tuple(mask.shape) != (len(dataset), candidates, 1)
                or not torch.all(mask[:, :count] == 1) or not torch.all(mask[:, count:] == 0)):
            raise ValueError("Dataset does not use the fixed nested sensor prefix")


def read_trial(study_dir: Path, config: dict, *, verify_checkpoints: bool = True) -> dict:
    directory = trial_dir(study_dir, config)
    record = read_json(directory / "result.json")
    unseal(record)
    if (record["schema"] != SCHEMA or record["status"] != "validation_complete"
            or record["configuration"] != config or record["test_evaluated"] is not False):
        raise RuntimeError("Trial identity/completion mismatch")
    if legacy.file_sha256(directory / "history.json") != record["history_sha256"]:
        raise RuntimeError("Trial history was changed")
    history = read_json(directory / "history.json")
    state = legacy.replay_early_stopping(config, history).state_dict()
    completion = legacy.completion_metadata(config, len(history), state)
    if any(record["training_result"][k] != v for k, v in completion.items()):
        raise RuntimeError("Trial completion differs from its validation history")
    for selection in legacy.CHECKPOINT_SELECTIONS:
        best = min(history, key=lambda row: row["validation"][selection])
        evaluated = record["evaluations"][selection]
        if (evaluated["selected_epoch"] != best["epoch"] or evaluated["validation_losses"] != best["validation"]
                or evaluated["selected_validation_loss"] != best["validation"][selection]):
            raise RuntimeError("Trial selection differs from the first validation minimum")
        if verify_checkpoints:
            path = directory / f"best_{selection}.pt"
            if legacy.file_sha256(path) != evaluated["checkpoint_sha256"]:
                raise RuntimeError("Selected checkpoint was changed")
    return record


def train_validation_run(study_dir: Path, config: dict, data: DevelopmentData, device: torch.device) -> dict:
    """Same v10 optimizer, epoch function, stopping and snapshots; zero test access."""
    if (study_dir / "selection.json").exists() or (study_dir / "final_evaluation_started.json").exists():
        raise RuntimeError("Selection is locked; training is no longer permitted in this study")
    validate_development(config, data)
    directory = trial_dir(study_dir, config)
    result_path = directory / "result.json"
    if result_path.exists():
        record = read_trial(study_dir, config)
        print(f"SKIP validation_complete: {directory.name}", flush=True)
        return record
    immutable_json(directory / "config.json", config)
    paths = legacy.run_checkpoint_paths(directory)
    settings = physics.TrainingSettings(**{key: config["training"][key]
                                           for key in physics.TrainingSettings.__dataclass_fields__})
    controls = legacy.regularization_from_config(config)
    tracker = legacy.DualObjectiveEarlyStopping(controls)
    weights = physics.LossWeights(**config["training"]["loss_weights"])
    seed = config["training"]["seed"]
    strict_seed(seed)
    model = legacy.model_from_config(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)
    loader = physics.create_data_loader(data.train, settings.batch_size, shuffle=True, seed=seed, device=device)
    val_loader = physics.create_data_loader(data.validation, settings.batch_size, device=device)
    best, history = {}, []
    elapsed_before = 0.0
    resumed = paths["latest"].exists()
    if resumed:
        latest = legacy.load_torch_checkpoint(paths["latest"], device)
        legacy.validate_resume_checkpoint(latest, config)
        model.load_state_dict(latest["model_state_dict"], strict=True)
        optimizer.load_state_dict(latest["optimizer_state_dict"])
        loader.generator.set_state(latest["shuffle_generator_state"])
        tracker.load_state_dict(latest["early_stopping"])
        history, best = latest["history"], latest["best_checkpoints"]
        elapsed_before = latest["elapsed_seconds"]
        legacy.restore_rng_state(latest["rng_state"])
        # latest.pt is the epoch transaction authority, including both best states.
        for selection in legacy.CHECKPOINT_SELECTIONS:
            legacy.atomic_torch_save(best[selection], paths[selection])
        legacy.atomic_write_json(directory / "history.json", history)
    elif ((directory / "history.json").exists()
          or any(paths[k].exists() for k in legacy.CHECKPOINT_SELECTIONS)):
        raise RuntimeError("Partial run without latest.pt; preserve it and use a new study")
    started = time.perf_counter()
    print(f"TRAIN {directory.name} | start={len(history)+1} max={settings.max_epochs} | test unused", flush=True)
    try:
        for epoch in range(len(history) + 1, settings.max_epochs + 1):
            if tracker.stopped:
                break
            train_loss = legacy.run_epoch(model, loader, optimizer, weights)
            val_loss = legacy.run_epoch(model, val_loader, weights=weights)
            history.append({"epoch": epoch, "train": asdict(train_loss), "validation": asdict(val_loss)})
            model_state = legacy.copy_model_state(model)
            changed = legacy.update_best_checkpoints(best, config=config, epoch=epoch,
                                                      validation=val_loss, model_state=model_state)
            tracker.update(epoch, val_loss)
            latest = legacy.make_resume_checkpoint(
                config=config, epoch=epoch, model_state=model_state, optimizer=optimizer,
                shuffle_generator=loader.generator, best=best, history=history,
                elapsed_seconds=elapsed_before + time.perf_counter() - started,
                early_stopping=tracker.state_dict())
            legacy.atomic_torch_save(latest, paths["latest"])
            for selection in changed:
                legacy.atomic_torch_save(best[selection], paths[selection])
            legacy.atomic_write_json(directory / "history.json", history)
            legacy.atomic_write_json(directory / "status.json",
                                     {"status": "training", "epoch": epoch, "updated_at": legacy.utc_now(),
                                      "run_fingerprint": digest(config), "early_stopping": tracker.state_dict()})
            if epoch == 1 or epoch % 10 == 0 or tracker.stopped or epoch == settings.max_epochs:
                print(f"  epoch={epoch:03d} train_structure={train_loss.structure:.6f} "
                      f"val_structure={val_loss.structure:.6f} val_total={val_loss.total:.6f} "
                      f"patience_used={tracker.bad_epochs}", flush=True)
        completion = legacy.completion_metadata(config, len(history), tracker.state_dict())
        evaluations = {}
        for selection in legacy.CHECKPOINT_SELECTIONS:
            selected = legacy.load_torch_checkpoint(paths[selection], device)
            legacy.validate_selected_checkpoint(selected, config, selection=selection,
                                                 expected_epoch=best[selection]["selected_epoch"],
                                                 expected_loss=best[selection]["selected_validation_loss"])
            model.load_state_dict(selected["model_state_dict"], strict=True)
            metrics = legacy.evaluate_model(model, data.validation, legacy.normalization_from_config(config),
                                             batch_size=settings.batch_size, weights=weights)
            if any(value is not None and not math.isfinite(value) for value in metrics.values()):
                raise FloatingPointError("Nonfinite validation diagnostic")
            evaluations[selection] = {
                "selected_epoch": selected["selected_epoch"],
                "selected_validation_loss": selected["selected_validation_loss"],
                "validation_losses": selected["validation_losses"], "validation_metrics": metrics,
                "checkpoint_sha256": legacy.file_sha256(paths[selection]),
                "checkpoint_path": paths[selection].relative_to(study_dir).as_posix()}
        legacy.atomic_write_csv(directory / "history.csv", [
            {"epoch": row["epoch"], **{f"{phase}_{k}": v for phase in ("train", "validation")
                                       for k, v in row[phase].items()}} for row in history])
        record = seal({
            "schema": SCHEMA, "status": "validation_complete", "test_evaluated": False,
            "configuration": config, "run_fingerprint": digest(config), "completed_at": legacy.utc_now(),
            "history_sha256": legacy.file_sha256(directory / "history.json"),
            "training_result": {**completion, "resumed": resumed,
                                "elapsed_seconds": elapsed_before + time.perf_counter() - started},
            "evaluations": evaluations})
        immutable_json(result_path, record)
        legacy.atomic_write_json(directory / "status.json",
                                 {"status": "validation_complete", "completed_at": record["completed_at"]})
        print(f"DONE {directory.name} | val_structure={evaluations['structure']['selected_validation_loss']:.6f} "
              f"| epochs={len(history)}", flush=True)
        return record
    except BaseException as error:
        legacy.atomic_write_json(directory / "status.json",
                                 {"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                                  "error": repr(error), "traceback": traceback.format_exc(),
                                  "updated_at": legacy.utc_now(), "last_committed_epoch":
                                  legacy.load_torch_checkpoint(paths["latest"], torch.device("cpu"))["epoch"]
                                  if paths["latest"].exists() else 0})
        raise


def score_candidates(study_dir: Path, study: dict, candidates: list[str], seeds: list[int]) -> list[dict]:
    rows = []
    for name in candidates:
        records = [read_trial(study_dir, config) for config in expected_trials(study, [name], seeds)]
        values = [record["evaluations"]["structure"]["selected_validation_loss"] for record in records]
        seed_scores = [float(np.mean([record["evaluations"]["structure"]["selected_validation_loss"]
                                     for record in records if record["configuration"]["training"]["seed"] == seed]))
                       for seed in seeds]
        rows.append({"candidate_id": name, "score": float(np.mean(values)),
                     "seed_std": legacy.sample_std(seed_scores), "seed_scores": dict(zip(map(str, seeds), seed_scores)),
                     "trial_count": len(records), "seeds": seeds,
                     "by_condition": {f"{model}/g{fraction:g}": float(np.mean([
                         record["evaluations"]["structure"]["selected_validation_loss"]
                         for record in records if record["configuration"]["model"]["name"] == model
                         and record["configuration"]["observation"]["g05_fraction"] == fraction]))
                         for model, fraction in itertools.product(study["spec"]["models"], study["spec"]["fractions"])}})
    return sorted(rows, key=lambda row: (row["score"], row["candidate_id"] != "baseline", row["candidate_id"]))


def promotion(study_dir: Path, study: dict) -> dict:
    ids = [candidate["id"] for candidate in study["candidates"]]
    rows = score_candidates(study_dir, study, ids, study["spec"]["screen_seeds"])
    promoted = ["baseline"] + [row["candidate_id"] for row in rows if row["candidate_id"] != "baseline"][:study["spec"]["top_k"]]
    authority = seal({"study_fingerprint": study["fingerprint"], "metric": "validation.structure",
                      "screening_ranking": rows, "promoted": promoted})
    immutable_json(study_dir / "promotion.json", authority)
    legacy.atomic_write_csv(study_dir / "screening.csv", rows)
    return authority


def choose_setting(rows: list[dict], spec: dict) -> tuple[str, list[dict], str]:
    baseline = next(row for row in rows if row["candidate_id"] == "baseline")
    annotated = []
    for row in rows:
        regressions = {key: 100 * (value - baseline["by_condition"][key]) / baseline["by_condition"][key]
                       for key, value in row["by_condition"].items()}
        annotated.append({**row, "improvement_pct": 100 * (baseline["score"] - row["score"]) / baseline["score"],
                          "condition_regression_pct": regressions,
                          "eligible": max(regressions.values()) <= spec["max_model_regression_pct"] + 1e-12})
    eligible = sorted((row for row in annotated if row["eligible"]),
                      key=lambda row: (row["score"], row["candidate_id"] != "baseline", row["candidate_id"]))
    best = eligible[0]
    if best["candidate_id"] != "baseline" and best["improvement_pct"] >= spec["min_improvement_pct"]:
        return best["candidate_id"], annotated, "lowest eligible mean validation structure; improvement threshold passed"
    return "baseline", annotated, "baseline retained: no eligible candidate cleared the predefined improvement threshold"


def lock_selection(study_dir: Path, study: dict) -> dict:
    if (study_dir / "selection.json").exists():
        return verify_selection(study_dir, study)
    promoted = promotion(study_dir, study)["promoted"]
    seeds = study["spec"]["screen_seeds"] + study["spec"]["confirmation_seeds"]
    ranking = score_candidates(study_dir, study, promoted, seeds)
    selected, annotated, reason = choose_setting(ranking, study["spec"])
    files = {}
    # Freeze all completed search results, not just a cherry-picked winning subset.
    for path in sorted((study_dir / "runs").glob("*/result.json")):
        record = read_json(path)
        read_trial(study_dir, record["configuration"])
        files[path.relative_to(study_dir).as_posix()] = legacy.file_sha256(path)
        history = path.with_name("history.json")
        files[history.relative_to(study_dir).as_posix()] = legacy.file_sha256(history)
    evaluation_runs = []
    for config in expected_trials(study, list(dict.fromkeys(["baseline", selected])), seeds):
        record = read_trial(study_dir, config)
        for selection in legacy.CHECKPOINT_SELECTIONS:
            evaluation = record["evaluations"][selection]
            files[evaluation["checkpoint_path"]] = evaluation["checkpoint_sha256"]
        evaluation_runs.append({"configuration": config, "trial": trial_dir(study_dir, config).name,
                                "result_path": (trial_dir(study_dir, config) / "result.json").relative_to(study_dir).as_posix()})
    selection = seal({"schema": SCHEMA, "study_fingerprint": study["fingerprint"],
                      "locked_at": legacy.utc_now(), "selected_candidate_id": selected,
                      "selected_hyperparameters": next(c for c in study["candidates"] if c["id"] == selected),
                      "reason": reason, "ranking": annotated, "seeds": seeds,
                      "promotion_sha256": legacy.file_sha256(study_dir / "promotion.json"),
                      "evaluation_runs": evaluation_runs, "artifact_sha256": files,
                      "test_used_for_selection": False})
    immutable_json(study_dir / "selection.json", selection)
    legacy.atomic_write_csv(study_dir / "confirmation.csv", annotated)
    print(f"LOCKED {selected}: {reason}", flush=True)
    return selection


def verify_selection(study_dir: Path, study: dict) -> dict:
    path = study_dir / "selection.json"
    if not path.exists():
        raise RuntimeError("Test access denied: run tuning and lock selection.json first")
    selected = read_json(path)
    unseal(selected)
    if (selected["study_fingerprint"] != study["fingerprint"]
            or selected["test_used_for_selection"] is not False
            or legacy.file_sha256(study_dir / "promotion.json") != selected["promotion_sha256"]):
        raise RuntimeError("Selection is not attached to the immutable study/promotion")
    for relative, expected in selected["artifact_sha256"].items():
        if legacy.file_sha256(study_dir / relative) != expected:
            raise RuntimeError(f"A frozen selection artifact was changed: {relative}")
    # Recompute ranking from validation-only records. No test scores are inspected.
    promoted = read_json(study_dir / "promotion.json")["promoted"]
    expected, ranking, reason = choose_setting(score_candidates(study_dir, study, promoted, selected["seeds"]), study["spec"])
    if expected != selected["selected_candidate_id"] or ranking != selected["ranking"] or reason != selected["reason"]:
        raise RuntimeError("Selection no longer matches the declared validation rule")
    return selected


def tune(study_dir: Path, study: dict, device: torch.device) -> dict:
    if (study_dir / "selection.json").exists():
        print("Selection already locked; no training or retuning performed", flush=True)
        return verify_selection(study_dir, study)
    if (study_dir / "final_evaluation_started.json").exists():
        raise RuntimeError("Cannot tune after test evaluation has started")
    data = development_data(study)
    for config in expected_trials(study, [c["id"] for c in study["candidates"]], study["spec"]["screen_seeds"]):
        train_validation_run(study_dir, config, data[config["observation"]["g05_fraction"]], device)
        refresh_trial_table(study_dir)
    promoted = promotion(study_dir, study)
    print("PROMOTED:", ", ".join(promoted["promoted"]), flush=True)
    for config in expected_trials(study, promoted["promoted"], study["spec"]["confirmation_seeds"]):
        train_validation_run(study_dir, config, data[config["observation"]["g05_fraction"]], device)
        refresh_trial_table(study_dir)
    return lock_selection(study_dir, study)


def fresh_holdout(study_dir: Path, study: dict, selection: dict) -> physics.DatasetArrays:
    """Created after the lock, with new charges but exactly the old sensor locations."""
    spec = study["spec"]["fresh_test"]
    path = study_dir / "final" / "fresh_holdout.npz"
    metadata_path = path.with_suffix(".json")
    if path.exists() and metadata_path.exists():
        metadata = read_json(metadata_path)
        if (metadata["selection_fingerprint"] != selection["fingerprint"]
                or metadata["sha256"] != legacy.file_sha256(path)):
            raise RuntimeError("Fresh holdout identity changed")
        return physics.load_dataset(path)
    original = physics.load_dataset(Path(study["data"]["path"]))
    generated = generator.generate_dataset(sample_count=spec["samples"], g05_point_count=original.g05.shape[1],
                                             seed=spec["seed"], charge_count=physics.CHARGE_COUNT)
    coordinates = original.g05[0, :, :2].astype(np.int64)
    x, y = original.grid_x[coordinates[:, 0]], original.grid_y[coordinates[:, 1]]
    for i, charges in enumerate(generated["target"].reshape(-1, physics.CHARGE_COUNT, 4)):
        q = charges.astype(np.float64)
        generated["G05"][i, :, :2] = coordinates
        generated["G05"][i, :, 2] = generator.coulomb_potential(x, y, q[:, 0], q[:, 1], q[:, 2], q[:, 3])
    generated["sensor_grid_indices"] = coordinates
    generated["sensor_policy"] = np.asarray("original training sensor coordinates; new independent charge seed")
    # If interrupted after NPZ publication, verify before publishing its manifest.
    if path.exists():
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(generated) or any(not np.array_equal(archive[k], v) for k, v in generated.items()):
                raise RuntimeError("Uncommitted fresh holdout differs from deterministic regeneration")
    else:
        legacy.atomic_save_npz(path, **generated)
    arrays = physics.load_dataset(path)
    old = {hashlib.sha256(row.tobytes()).hexdigest() for row in original.g00}
    new = {hashlib.sha256(row.tobytes()).hexdigest() for row in arrays.g00}
    if old & new or len(new) != len(arrays.g00):
        raise RuntimeError("Fresh holdout contains duplicate/development samples")
    immutable_json(metadata_path, {"selection_fingerprint": selection["fingerprint"],
                                   "sha256": legacy.file_sha256(path), "array_hashes": dataset_hashes(arrays),
                                   "samples": spec["samples"], "seed": spec["seed"],
                                   "original_sample_overlap": 0, "same_sensor_coordinates": True})
    return arrays


def finalize(study_dir: Path, study: dict, device: torch.device) -> dict:
    # Gate is checked BEFORE opening/preparing/generating any test data.
    selection = verify_selection(study_dir, study)
    started_path = study_dir / "final_evaluation_started.json"
    if started_path.exists():
        marker = read_json(started_path)
        if marker["selection_fingerprint"] != selection["fingerprint"]:
            raise RuntimeError("A different selection already opened the holdouts")
    else:
        immutable_json(started_path, {"selection_fingerprint": selection["fingerprint"],
                                     "started_at": legacy.utc_now(), "hyperparameter_search_closed": True})
    complete_path = study_dir / "final" / "result.json"
    if complete_path.exists():
        complete = read_json(complete_path)
        unseal(complete)
        if complete["selection_fingerprint"] != selection["fingerprint"]:
            raise RuntimeError("Final result belongs to a different selection")
        for path, sha in complete["artifact_sha256"].items():
            if legacy.file_sha256(study_dir / path) != sha:
                raise RuntimeError(f"Final evaluation artifact changed: {path}")
        print("Final evaluation already complete; returning saved results without opening test", flush=True)
        return complete
    original = physics.load_dataset(Path(study["data"]["path"]))
    fresh = fresh_holdout(study_dir, study, selection)
    stats = legacy.normalization_from_config(study)
    records = []
    for split_name, arrays, indices in (
            ("historical_test", original, np.asarray(study["split"]["test"])),
            ("fresh_test", fresh, np.arange(len(fresh.target)))):
        for fraction in study["spec"]["fractions"]:
            dataset = physics.prepare_dataset(arrays, indices, stats, fraction)
            for item in selection["evaluation_runs"]:
                config = item["configuration"]
                if config["observation"]["g05_fraction"] != fraction:
                    continue
                for objective in legacy.CHECKPOINT_SELECTIONS:
                    output_path = study_dir / "final" / "evaluations" / f"{split_name}__{item['trial']}__{objective}.json"
                    validation = read_trial(study_dir, config)["evaluations"][objective]
                    identity = {"selection_fingerprint": selection["fingerprint"], "split": split_name,
                                "trial": item["trial"], "candidate_id": config["tuning"]["candidate_id"],
                                "model": config["model"]["name"], "fraction": fraction,
                                "seed": config["training"]["seed"], "checkpoint_selection": objective,
                                "checkpoint_sha256": validation["checkpoint_sha256"],
                                "selected_epoch": validation["selected_epoch"],
                                "sample_count": len(dataset)}
                    if output_path.exists():
                        record = read_json(output_path)
                        unseal(record)
                        if any(record.get(k) != v for k, v in identity.items()):
                            raise RuntimeError("Interrupted evaluation has a different identity")
                    else:
                        model, loaded_stats, _ = load_trained_model(study_dir / validation["checkpoint_path"], device)
                        losses = legacy.run_epoch(model, physics.create_data_loader(
                            dataset, study["spec"]["batch_size"], device=device))
                        metrics = legacy.evaluate_model(model, dataset, loaded_stats,
                                                        batch_size=study["spec"]["batch_size"])
                        if any(v is not None and not math.isfinite(v) for v in metrics.values()):
                            raise FloatingPointError("Nonfinite final diagnostic")
                        record = seal({**identity, "evaluated_at": legacy.utc_now(),
                                       "losses": asdict(losses), "metrics": metrics})
                        immutable_json(output_path, record)
                        print(f"FINAL {split_name} {item['trial']} {objective} | "
                              f"structure={losses.structure:.6f} position_mae={metrics['mean_position_mae']:.6f}",
                              flush=True)
                    records.append(record)
    artifacts = {p.relative_to(study_dir).as_posix(): legacy.file_sha256(p)
                 for p in sorted((study_dir / "final").glob("evaluations/*.json"))}
    for name in ("fresh_holdout.npz", "fresh_holdout.json"):
        path = study_dir / "final" / name
        artifacts[path.relative_to(study_dir).as_posix()] = legacy.file_sha256(path)
    final = seal({"schema": SCHEMA, "selection_fingerprint": selection["fingerprint"],
                  "completed_at": legacy.utc_now(), "records": records, "artifact_sha256": artifacts,
                  "note": "Frozen selection; no test-driven tuning or checkpoint switching. Fresh holdout is IID simulation only."})
    immutable_json(complete_path, final)
    return final


def refresh_trial_table(study_dir: Path) -> list[dict]:
    rows = []
    for path in sorted((study_dir / "runs").glob("*/result.json")):
        record = read_json(path)
        unseal(record)
        config = record["configuration"]
        for selection, evaluation in record["evaluations"].items():
            rows.append({"trial": path.parent.name, "candidate_id": config["tuning"]["candidate_id"],
                         "model": config["model"]["name"], "fraction": config["observation"]["g05_fraction"],
                         "seed": config["training"]["seed"], "checkpoint_selection": selection,
                         "learning_rate": config["training"]["learning_rate"],
                         "weight_decay": config["training"]["weight_decay"],
                         "structure_dropout": config["training"]["regularization"]["structure_dropout"],
                         "epochs_completed": record["training_result"]["epochs_completed"],
                         "stop_reason": record["training_result"]["stop_reason"],
                         "elapsed_seconds": record["training_result"]["elapsed_seconds"],
                         "selected_epoch": evaluation["selected_epoch"],
                         **{f"val_loss_{k}": v for k, v in evaluation["validation_losses"].items()},
                         **{f"val_{k}": v for k, v in evaluation["validation_metrics"].items()}})
    if rows:
        legacy.atomic_write_csv(study_dir / "trials.csv", rows)
    return rows


def comparison_tables(study_dir: Path, study: dict, selected: dict, final: dict | None):
    observations = []
    for item in selected["evaluation_runs"]:
        record = read_trial(study_dir, item["configuration"])
        config = item["configuration"]
        for objective, evaluation in record["evaluations"].items():
            observations.append({"split": "validation", "candidate_id": config["tuning"]["candidate_id"],
                                 "model": config["model"]["name"], "fraction": config["observation"]["g05_fraction"],
                                 "seed": config["training"]["seed"], "checkpoint_selection": objective,
                                 "values": {**{f"loss_{k}": v for k, v in evaluation["validation_losses"].items()},
                                            **evaluation["validation_metrics"]}})
    if final:
        for record in final["records"]:
            observations.append({**{key: record[key] for key in
                                    ("split", "candidate_id", "model", "fraction", "seed", "checkpoint_selection")},
                                 "values": {**{f"loss_{k}": v for k, v in record["losses"].items()},
                                            **record["metrics"]}})
    rows = []
    selected_id = selected["selected_candidate_id"]
    index = {(r["split"], r["candidate_id"], r["model"], r["fraction"], r["seed"], r["checkpoint_selection"]): r
             for r in observations}
    for key, baseline in index.items():
        split_name, candidate, model, fraction, seed, objective = key
        if candidate != "baseline":
            continue
        tuned = index[(split_name, selected_id, model, fraction, seed, objective)]
        for metric, before in baseline["values"].items():
            after = tuned["values"][metric]
            lower = not metric.endswith("accuracy")
            delta = None if before is None or after is None else (before - after if lower else after - before)
            rows.append({"split": split_name, "model": model, "fraction": fraction, "seed": seed,
                         "checkpoint_selection": objective, "metric": metric, "selected_candidate_id": selected_id,
                         "baseline": before, "selected": after, "improvement": delta,
                         "improvement_pct": None if delta is None or before == 0 else 100 * delta / abs(before),
                         "accuracy_delta_pp": 100 * delta if delta is not None and metric.endswith("accuracy") else None})
    legacy.atomic_write_csv(study_dir / "paired_comparisons.csv", rows)
    summary = []
    group_keys = ("split", "model", "fraction", "checkpoint_selection", "metric")
    for key in sorted({tuple(row[k] for k in group_keys) for row in rows}):
        group = [row for row in rows if tuple(row[k] for k in group_keys) == key]
        valid = [row for row in group if row["improvement"] is not None]
        result = dict(zip(group_keys, key))
        for field in ("baseline", "selected", "improvement"):
            values = [row[field] for row in valid]
            result[f"{field}_mean"] = float(np.mean(values)) if values else None
            result[f"{field}_std"] = legacy.sample_std(values)
        before, delta = result["baseline_mean"], result["improvement_mean"]
        result.update({"paired_seeds": len(valid), "improved_seeds": sum(r["improvement"] > 0 for r in valid),
                       "improvement_pct": 100 * delta / abs(before) if before else None})
        summary.append(result)
    legacy.atomic_write_csv(study_dir / "comparison_summary.csv", summary)
    routing = []
    for observation in observations:
        if observation["model"] != "g05_sign_only":
            continue
        full = index[(observation["split"], observation["candidate_id"], "g05_full_reconstruction",
                      observation["fraction"], observation["seed"], observation["checkpoint_selection"])]
        for metric, sign in observation["values"].items():
            value = full["values"][metric]
            routing.append({k: observation[k] for k in ("split", "candidate_id", "fraction", "seed", "checkpoint_selection")}
                           | {"metric": metric, "sign_only": sign, "full": value,
                              "full_improvement": None if sign is None or value is None
                              else (value - sign if metric.endswith("accuracy") else sign - value)})
    legacy.atomic_write_csv(study_dir / "routing_comparisons.csv", routing)
    return summary


def report(study_dir: Path, study: dict) -> None:
    rows = refresh_trial_table(study_dir)
    if not (study_dir / "selection.json").exists():
        print(f"{len(rows)//2} validation-complete trials; selection not yet locked", flush=True)
        return
    selected = verify_selection(study_dir, study)
    final_path = study_dir / "final" / "result.json"
    final = read_json(final_path) if final_path.exists() else None
    if final:
        unseal(final)
        if final["selection_fingerprint"] != selected["fingerprint"]:
            raise RuntimeError("Report cannot mix selections")
        for relative, sha in final["artifact_sha256"].items():
            if legacy.file_sha256(study_dir / relative) != sha:
                raise RuntimeError("Final report source changed")
    summary = comparison_tables(study_dir, study, selected, final)
    baseline = next(row for row in selected["ranking"] if row["candidate_id"] == "baseline")
    winner = next(row for row in selected["ranking"] if row["candidate_id"] == selected["selected_candidate_id"])
    lines = [
        "# ModelExperiment11 experiment report", "",
        f"Selected common setting: **{selected['selected_candidate_id']}**. {selected['reason']}.", "",
        f"Primary validation structure: **{baseline['score']:.6f} -> {winner['score']:.6f}** "
        f"({winner['improvement_pct']:+.2f}% improvement).",
        "This is a selection-set estimate; it is not an unbiased estimate of generalization.", "",
        f"Hyperparameters: {json.dumps(selected['selected_hyperparameters'], sort_keys=True)}", "",
        f"Completed training trials: {len(rows)//2}. "
        f"Search candidates: {len(study['candidates'])}. Seeds: {selected['seeds']}. "
        f"Fractions: {study['spec']['fractions']}.",
        f"Train / validation / historical test: {study['split_audit']['counts']}. "
        f"Split seed: {study['split_seed']}.",
        "The unchanged full and sign-only architectures share one selected setting, fixed loss weights, "
        "sensor prefixes, initialization/shuffle seeds, batch size, and maximum epoch budget.", "",
        "| Candidate | Validation structure | Seed SD | Improvement | Eligible |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in selected["ranking"]:
        sd = "N/A" if row["seed_std"] is None else f"{row['seed_std']:.6f}"
        lines.append(f"| {row['candidate_id']} | {row['score']:.6f} | {sd} | {row['improvement_pct']:+.2f}% | {row['eligible']} |")
    lines += ["", "Primary checkpoint (best_structure.pt); positive improvement is better:", "",
              "| Split | Model | Fraction | Metric | Baseline | Selected | Improvement |",
              "| --- | --- | ---: | --- | ---: | ---: | ---: |"]
    shown = {"loss_structure", "mean_position_mae", "mean_position_3d_error", "relative_sign_accuracy",
             "global_sign_accuracy", "absolute_sign_accuracy"}
    for row in summary:
        if row["checkpoint_selection"] != "structure" or row["metric"] not in shown:
            continue
        before, after, delta = (row[k] for k in ("baseline_mean", "selected_mean", "improvement_mean"))
        if before is None:
            formatted = ("N/A", "N/A", "N/A")
        elif row["metric"].endswith("accuracy"):
            formatted = (f"{100*before:.2f}%", f"{100*after:.2f}%", f"{100*delta:+.2f} pp")
        else:
            formatted = (f"{before:.6f}", f"{after:.6f}", f"{row['improvement_pct']:+.2f}%")
        lines.append(f"| {row['split']} | {row['model']} | {row['fraction']:g} | {row['metric']} | "
                     + " | ".join(formatted) + " |")
    lines += [
        "", "Full paired values, sample SD across seeds (ddof=1), losses and both checkpoint selections are in "
        "paired_comparisons.csv / comparison_summary.csv. routing_comparisons.csv retains the original "
        "full versus sign-only comparison under identical settings.", "",
        "Test status: " + ("completed after the immutable selection lock." if final else "unopened in this study."),
        "Historical test was examined in earlier work. The fresh holdout uses the preregistered new simulation "
        "seed and the original sensor coordinates; it is generated only after selection.",
        "No train+validation refit or test-based seed/checkpoint selection is performed.",
        "The same validation split is used for epoch and hyperparameter selection. Three training seeds measure "
        "initialization/order variation, not three independent data splits. Search is bounded, not globally optimal. "
        "No evidence is claimed for unsearched sensor fractions, other sensor layouts, noisy measurements, or real data.",
        "Position and sign metrics use the unchanged joint 120-permutation assignment. Their values are not "
        "position-only matching diagnostics. Lower composite loss does not imply every component improves.", "",
        "Reproduction requires the saved source/data hashes, study configuration, runtime, and device. "
        "Exact interrupted-resume equality is scoped to that environment. "
        "[PyTorch reproducibility](https://docs.pytorch.org/docs/2.7/notes/randomness.html). "
        "Seeded log-uniform random candidates supplement evidence-based anchors; the general motivation "
        "for bounded random search is [Bergstra & Bengio (2012)](https://www.jmlr.org/papers/v13/bergstra12a.html).",
    ]
    text = "\n".join(lines) + "\n"
    (study_dir / "report.md").write_text(text, encoding="utf-8")
    print(f"REPORT {study_dir / 'report.md'}", flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", nargs="?", choices=("init", "tune", "select", "finalize", "report", "run"), default="run")
    parser.add_argument("--study-dir", type=Path, default=DEFAULT_STUDY)
    parser.add_argument("--config", type=Path, default=None, help="New study configuration; immutable after initialization")
    parser.add_argument("--data", type=Path, default=None, help="Five-charge NPZ for a new study")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but not available")
    study_dir = args.study_dir.resolve()
    # Protect old results/checkpoints even if a mistaken CLI path names them.
    for protected in ("Codes", "Models", "Results", "Documents", ".git", ".agents", ".codex"):
        if study_dir == ROOT / protected or ROOT / protected in study_dir.parents:
            parser.error("Study output must be separate from all legacy artifact/code directories")
    if study_dir == ROOT:
        parser.error("Use a dedicated empty study directory")
    with legacy.experiment_locks(study_dir):
        if (study_dir / "study.json").exists():
            study = load_study(study_dir, device=device if args.command != "report" else None)
            if args.config is not None and read_json(args.config) != study["spec"]:
                parser.error("Configuration differs from the frozen study; use a new --study-dir")
            if args.data is not None and args.data.resolve() != Path(study["data"]["path"]).resolve():
                parser.error("Data path differs from the frozen study")
        elif args.command in ("init", "tune", "run"):
            study = initialize(study_dir, read_json(args.config or DEFAULT_CONFIG),
                               args.data or physics.DEFAULT_DATA_PATH, device)
        else:
            parser.error("Study does not exist; initialize it first")
        if args.command == "init":
            return
        if args.command in ("tune", "run"):
            tune(study_dir, study, device)
        if args.command == "select":
            lock_selection(study_dir, study)
        if args.command in ("finalize", "run"):
            finalize(study_dir, study, device)
        report(study_dir, study)


if __name__ == "__main__":
    main()
