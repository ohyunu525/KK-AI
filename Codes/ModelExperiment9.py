"""Five-charge G05 routing ablation, preserving the NewLearning9 physics protocol.

python Codes/ModelExperiment9.py --fractions 0.75 --seeds 42 --epochs 300 --smoke-only
Remove --smoke-only to train the two capacity-matched models.
Use --evaluate-only with an existing --experiment-name to evaluate saved checkpoints without training.
NewLearning9.py remains the authority for data, exact set assignment, losses and metrics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import platform
import random
import re
import sys
import tempfile
import time
import traceback
import uuid
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

# Import the baseline first: it configures cuBLAS before CUDA matrix operations.
import NewLearning9 as physics
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = physics.DEFAULT_DATA_PATH
DEFAULT_RESULTS_ROOT = PROJECT_DIR / "Results" / "new_learning9_experiments"
DEFAULT_CHECKPOINT_ROOT = PROJECT_DIR / "Models" / "new_learning9_experiments"
DEFAULT_EXPERIMENT_NAME = "g05_routing_v1"
DEFAULT_MODELS = ("g05_sign_only", "g05_full_reconstruction")
PROTOCOL_VERSION = "new-learning9-g05-routing-v1"
CHECKPOINT_SCHEMA_VERSION = 1
RESULT_SCHEMA_VERSION = 1
CHECKPOINT_SELECTIONS = ("total", "structure")
STRUCTURE_METRIC_NAMES = (
    "position_mae_x", "position_mae_y", "position_mae_z", "mean_position_mae",
    "mean_position_3d_error", "charge_magnitude_mae", "relative_sign_accuracy",
    "relative_configuration_accuracy", "pairwise_relative_sign_accuracy",
)
GLOBAL_METRIC_NAMES = (
    "global_sign_accuracy", "global_sign_bce", "absolute_sign_accuracy",
    "absolute_sign_set_accuracy", "charge_mae", "global_invariant_charge_mae",
)
METRIC_NAMES = STRUCTURE_METRIC_NAMES + GLOBAL_METRIC_NAMES
LOWER_IS_BETTER = frozenset(name for name in METRIC_NAMES if not name.endswith("accuracy"))
STRUCTURE_PREFIXES = physics.STRUCTURE_PREFIXES + ("structure_context.",)
OUTPUT_FIELDS = ("position", "magnitude", "relative_sign_logit", "global_sign_logit")
TrainingSettings = physics.TrainingSettings
LossWeights = physics.LossWeights
EpochLoss = physics.EpochLoss
run_epoch = physics.run_epoch
evaluate_model = physics.evaluate_model
copy_model_state = physics.copy_state


class RoutedChargeNet(physics.ChargeNet):
    """Identical modules and initialization; only the G05-to-structure route differs."""

    def __init__(self, *, allow_g05_for_structure: bool) -> None:
        # Keep every original module and its construction order, including G05.
        super().__init__()
        self.allow_g05_for_structure = allow_g05_for_structure
        self.structure_context = nn.Sequential(
            nn.Linear(32 * 3, 128), nn.ReLU(), nn.Linear(128, 256),
        )
        nn.init.zeros_(self.structure_context[-1].weight)
        nn.init.zeros_(self.structure_context[-1].bias)

    def forward(self, g00: torch.Tensor, g05: torch.Tensor, g05_mask: torch.Tensor) -> physics.ModelOutput:
        if not self.allow_g05_for_structure:
            return super().forward(g00, g05, g05_mask)
        if (g05.ndim != 3 or g05.shape[-1] != 3 or g05.shape[1] < 1
                or g05_mask.shape != (*g05.shape[:2], 1) or g05.shape[0] != g00.shape[0]):
            raise ValueError(f"G00/G05/mask shape mismatch: {g00.shape}, {g05.shape}, {g05_mask.shape}")
        structure = self.g00_encoder(self.g00_cnn(g00))
        # The inherited G05-only odd function is intentionally unchanged.
        global_logit = self.forward_global_sign(g05, g05_mask)
        observed = g05_mask.sum(dim=(1, 2)) > 0
        if torch.any(observed):
            points = g05.masked_fill(~g05_mask.bool(), 0)
            reversed_points = points * points.new_tensor((1, 1, -1))
            summaries = self._masked_summary(
                self.g05_encoder(torch.cat((points, reversed_points), dim=0)),
                torch.cat((g05_mask, g05_mask), dim=0),
            )
            positive, negative = summaries.chunk(2)
            # Structure must be even under a whole-set sign reversal; global sign is odd.
            even_summary = (positive + negative) * 0.5
            context = self.structure_context(even_summary) * observed[:, None]
            structure = structure + context
        return physics.ModelOutput(
            position=self.position_head(structure).reshape(-1, physics.CHARGE_COUNT, 3),
            magnitude=F.softplus(self.magnitude_head(structure)),
            relative_sign_logit=self.relative_sign_head(structure),
            global_sign_logit=global_logit,
        )


@dataclass(frozen=True)
class ModelSpec:
    name: str
    allow_g05_for_structure: bool

    def factory(self) -> RoutedChargeNet:
        return RoutedChargeNet(allow_g05_for_structure=self.allow_g05_for_structure)

    @property
    def input_policy(self) -> dict[str, str]:
        structure_input = "G00 + masked even G05 summary" if self.allow_g05_for_structure else "G00 only"
        return {
            "position": structure_input, "magnitude": structure_input,
            "relative_sign": structure_input, "global_sign": "masked G05 only; odd in V",
        }


MODEL_REGISTRY = {
    name: ModelSpec(name, name == "g05_full_reconstruction") for name in DEFAULT_MODELS
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_ready(item) for item in value]
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(
        json_ready(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def object_fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def replace_with_retry(source: Path, destination: Path) -> None:
    """Atomically replace a file despite short-lived Windows file locks."""

    for attempt in range(8):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(min(0.05 * (2**attempt), 0.8))


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                json_ready(value),
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        replace_with_retry(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def atomic_write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows and not path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    if not rows:
        # Clear stale rows without changing an existing report's columns.
        with path.open("r", encoding="utf-8", newline="") as handle:
            fieldnames = next(csv.reader(handle), [])
    try:
        with temporary_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(
                {
                    key: "" if value is None else value
                    for key, value in json_ready(row).items()
                }
                for row in rows
            )
            handle.flush()
            os.fsync(handle.fileno())
        replace_with_retry(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary_path.open("wb") as handle:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        replace_with_retry(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def atomic_torch_save(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary_path.open("wb") as handle:
            torch.save(dict(value), handle)
            handle.flush()
            os.fsync(handle.fileno())
        replace_with_retry(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_torch_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    # Checkpoints also contain CPU-only RNG state tensors.  Loading the whole
    # mapping onto CUDA makes Generator.set_state()/torch.set_rng_state() fail.
    # Model parameters are copied by load_state_dict; AdamW.load_state_dict
    # restores moment buffers to parameter devices while keeping CPU step counters.
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint is not a mapping: {path}")
    return checkpoint


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def parameter_counts(model: nn.Module) -> dict[str, int]:
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }


def sample_std(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=1))


def has_nonzero_gradient(model: nn.Module, prefixes: tuple[str, ...]) -> bool:
    return any(
        name.startswith(prefixes)
        and parameter.grad is not None
        and torch.any(parameter.grad != 0).item()
        for name, parameter in model.named_parameters()
    )


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    physics.set_reproducibility(seed)


def selection_policy(selection: str, *, g05_count: int | None, global_sign_weight: float) -> dict[str, Any]:
    if selection not in CHECKPOINT_SELECTIONS:
        raise ValueError(f"Unknown checkpoint selection: {selection}")
    includes_global: bool | None = False
    if selection == "total" and global_sign_weight != 0:
        includes_global = None if g05_count is None else g05_count > 0
    return {
        "checkpoint_selection": selection,
        "selection_objective": f"validation_loss.{selection}",
        "selection_note": "One complete epoch state; first epoch wins ties; no component composition",
        "primary_metrics": list(STRUCTURE_METRIC_NAMES if selection == "structure" else METRIC_NAMES),
        "global_sign_in_selection_objective": includes_global,
        "global_sign_metrics_note": (
            "Global-sign and absolute-sign metrics are secondary diagnostics: structure selection "
            "does not optimize global-sign performance; training still uses the unchanged total loss."
            if selection == "structure" else
            "Total selection includes global-sign loss only with observed G05 and nonzero global-sign weight."
        ),
    }


def runtime_environment(device: torch.device) -> dict[str, Any]:
    return {
        "python": platform.python_version(), "numpy": np.__version__,
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(), "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "platform": platform.platform(), "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "torch_num_threads": torch.get_num_threads(), "torch_num_interop_threads": torch.get_num_interop_threads(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "deterministic_algorithms": "enabled, warn_only=True (unchanged baseline)",
    }


def build_protocol(
    *, data_path: Path, arrays: physics.DatasetArrays, split: physics.DataSplit,
    stats: physics.NormalizationStats, settings: TrainingSettings,
    weights: LossWeights = LossWeights(), device: torch.device,
) -> dict[str, Any]:
    baseline = physics.checkpoint_metadata(arrays, stats, split, data_path)
    # These describe the baseline's routing/composition, not its physical protocol.
    excluded = {"protocol_version", "model_architecture", "input_policy", "gradient_policy", "checkpoint_policy"}
    return json_ready({
        "protocol_version": PROTOCOL_VERSION,
        "baseline_protocol_version": physics.PROTOCOL_VERSION,
        "physics": {key: value for key, value in baseline.items() if key not in excluded},
        "data": {"path": str(data_path.resolve()), "sha256": file_sha256(data_path),
                 "sample_count": len(arrays.target), "g00_shape": arrays.g00.shape,
                 "g05_shape": arrays.g05.shape, "target_shape": arrays.target.shape,
                 "candidate_count": arrays.g05.shape[1]},
        "source_sha256": {"ModelExperiment9.py": file_sha256(Path(__file__)),
                          "NewLearning9.py": file_sha256(Path(physics.__file__))},
        "normalization": stats.to_dict(),
        "training": {**asdict(settings), "optimizer": "AdamW", "loss_weights": asdict(weights),
                     "structure_reuse": False, "joint_gradient_clipping": False,
                     "checkpoint_selection": {
                         selection: selection_policy(selection, g05_count=None, global_sign_weight=weights.global_sign)
                         for selection in CHECKPOINT_SELECTIONS}},
        "models": {name: {"input_policy": spec.input_policy, "capacity_matched": True}
                   for name, spec in MODEL_REGISTRY.items()},
        "fusion": {"method": "G00 feature + observed-mask * structure_context(mean(summary(V), summary(-V)))",
                   "g05_summary": "unchanged masked mean/max/std; same g05_encoder as global sign",
                   "structure_feature_size": 256, "context_hidden_size": 128,
                   "final_projection_zero_initialized": True,
                   "shared_encoder_note": "Full model G05 encoder receives both structure and global-sign gradients; "
                                          "global loss has no path into G00, structure heads or structure_context"},
        "evaluation": {"metrics": METRIC_NAMES, "structure_primary_metrics": STRUCTURE_METRIC_NAMES,
                       "symmetry_validation": "exact algebraic symmetry; float32 checks allow atol=1e-7, rtol=1e-6 (baseline roundoff)",
                       "test_used_only_after_training": True, "same_test_set_for_both_selections": True,
                       "checkpoint_policy": "independent minima of validation structure and total; whole epoch states",
                       "pairing": "same protocol, fraction, seed and checkpoint selection",
                       "positive_improvement": "g05_full_reconstruction is better",
                       "standard_deviation": "sample std ddof=1; N/A with fewer than two seeds"},
        "environment": runtime_environment(device),
    })


@contextmanager
def experiment_locks(*roots: Path):
    """Reject concurrent CLI writers; OS locks are released even after process death."""
    with ExitStack() as handles:
        for root in sorted({path.resolve() for path in roots}, key=lambda path: str(path).casefold()):
            # Sibling lock files do not change a legacy or incompatible experiment directory.
            path = root.with_name(f".{root.name}.lock")
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = handles.enter_context(path.open("a+b"))
            if os.fstat(handle.fileno()).st_size == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            try:
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                raise RuntimeError(f"Experiment is already running or locked; stop it before resuming: {root}") from error
        # Closing the handles releases the locks; their files may safely remain.
        yield


def initialize_experiment_artifacts(
    *, experiment_results_dir: Path, experiment_checkpoint_dir: Path,
    protocol: Mapping[str, Any], split: physics.DataSplit,
) -> str:
    fingerprint = object_fingerprint(protocol)
    path = experiment_results_dir / "protocol.json"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        payload = {key: value for key, value in saved.items() if key != "protocol_fingerprint"}
        if saved.get("protocol_fingerprint") != fingerprint or object_fingerprint(payload) != fingerprint:
            raise RuntimeError("Experiment protocol changed; keep old artifacts and use a new --experiment-name")
    else:
        for root in (experiment_results_dir, experiment_checkpoint_dir):
            if root.exists() and any(root.iterdir()):
                raise RuntimeError(f"Nonempty directory without a matching protocol; use a new --experiment-name: {root}")
        atomic_write_json(path, {**protocol, "protocol_fingerprint": fingerprint})
    experiment_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    normalization_path = experiment_results_dir / "normalization.json"
    if normalization_path.exists():
        if canonical_json(json.loads(normalization_path.read_text(encoding="utf-8"))) != canonical_json(protocol["normalization"]):
            raise RuntimeError("Saved normalization does not match the experiment protocol")
    else:
        atomic_write_json(normalization_path, protocol["normalization"])
    split_path = experiment_results_dir / "split_indices.npz"
    if split_path.exists():
        with np.load(split_path, allow_pickle=False) as saved_split:
            if any(name not in saved_split or not np.array_equal(saved_split[name], getattr(split, name))
                   for name in ("train", "validation", "test")):
                raise RuntimeError("Saved split does not match the experiment protocol")
    else:
        atomic_save_npz(split_path, **{name: getattr(split, name) for name in ("train", "validation", "test")})
    return fingerprint


def run_configuration(protocol: Mapping[str, Any], *, model_name: str, fraction: float, seed: int) -> dict[str, Any]:
    spec = MODEL_REGISTRY[model_name]
    candidate_count = int(protocol["data"]["candidate_count"])
    count = physics.g05_count_for_fraction(fraction, candidate_count)
    with torch.random.fork_rng(devices=[]):
        prototype = spec.factory()
        state_shapes = {name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
                        for name, tensor in prototype.state_dict().items()}
        counts = parameter_counts(prototype)
    training = {key: protocol["training"][key] for key in TrainingSettings.__dataclass_fields__}
    weights = protocol["training"]["loss_weights"]
    return json_ready({
        "protocol_version": PROTOCOL_VERSION, "protocol_fingerprint": object_fingerprint(protocol),
        "charge_count": physics.CHARGE_COUNT,
        "model": {"name": model_name, "input_policy": spec.input_policy,
                  "parameter_count": counts, "state_shapes": state_shapes},
        "observation": {"g05_fraction": fraction, "g05_count_per_sample": count,
                        "candidate_count": candidate_count, "selection": "fixed nested sensor prefix",
                        "full_fraction_definition": "all stored candidate sensors, not the full 32x32 field"},
        "training": {**training, "seed": seed, "optimizer": "AdamW", "loss_weights": weights,
                     "checkpoint_selection": {
                         selection: selection_policy(selection, g05_count=count, global_sign_weight=weights["global_sign"])
                         for selection in CHECKPOINT_SELECTIONS}},
        "normalization": protocol["normalization"],
        "split_counts": {name: len(protocol["physics"]["split_indices"][name]) for name in ("train", "validation", "test")},
    })


def run_id_for(config: Mapping[str, Any]) -> str:
    return (f"{config['model']['name']}__g05_{round(config['observation']['g05_fraction'] * 100):03d}pct"
            f"__seed_{config['training']['seed']}__{object_fingerprint(config)[:12]}")


def run_metadata(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION, "protocol_fingerprint": config["protocol_fingerprint"],
        "run_fingerprint": object_fingerprint(config), "run_id": run_id_for(config),
        "model_name": config["model"]["name"], "charge_count": physics.CHARGE_COUNT,
        "g05_fraction": config["observation"]["g05_fraction"],
        "g05_count_per_sample": config["observation"]["g05_count_per_sample"], "seed": config["training"]["seed"],
    }


def run_checkpoint_paths(checkpoint_dir: Path) -> dict[str, Path]:
    return {"latest": checkpoint_dir / "latest.pt",
            **{selection: checkpoint_dir / f"best_{selection}.pt" for selection in CHECKPOINT_SELECTIONS}}


def normalization_from_config(config: Mapping[str, Any]) -> physics.NormalizationStats:
    values = dict(config["normalization"])
    for name in ("position_mean", "position_std"):
        values[name] = np.asarray(values[name], dtype=np.float32)
    return physics.NormalizationStats(**values)


def validate_model_state(state: Any, config: Mapping[str, Any]) -> None:
    expected = config["model"]["state_shapes"]
    if not isinstance(state, Mapping) or set(state) != set(expected):
        raise RuntimeError("Checkpoint must contain the complete model state, not a branch/component")
    for name, spec in expected.items():
        value = state[name]
        if (not isinstance(value, torch.Tensor) or list(value.shape) != spec["shape"]
                or str(value.dtype) != spec["dtype"] or not torch.isfinite(value).all()):
            raise RuntimeError(f"Invalid model state tensor: {name}")


def validate_identity(value: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    if config.get("protocol_version") != PROTOCOL_VERSION or config.get("charge_count") != physics.CHARGE_COUNT:
        raise RuntimeError("Configuration is not for this five-charge routing experiment")
    for key, expected in run_metadata(config).items():
        if value.get(key) != expected:
            raise RuntimeError(f"Run metadata mismatch: {key}")
    if canonical_json(value.get("configuration")) != canonical_json(config):
        raise RuntimeError("Run configuration mismatch")


def best_tracking_fields(best: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {f"best_{selection}_{field}": best.get(selection, {}).get(source)
            for selection in CHECKPOINT_SELECTIONS
            for field, source in (("loss", "selected_validation_loss"), ("epoch", "selected_epoch"))}


def validate_loss_values(losses: Any, config: Mapping[str, Any], label: str) -> None:
    if not isinstance(losses, Mapping) or set(losses) != set(EpochLoss.__dataclass_fields__):
        raise RuntimeError(f"Incomplete {label}")
    observed = config["observation"]["g05_count_per_sample"] > 0
    for name, value in losses.items():
        if name == "global_sign" and not observed:
            if value is not None:
                raise RuntimeError(f"Unobserved global sign must be N/A in {label}")
        elif value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise RuntimeError(f"Non-finite or missing {name} in {label}")


def update_best_checkpoints(
    best: dict[str, dict[str, Any]], *, config: Mapping[str, Any], epoch: int,
    validation: EpochLoss, model_state: dict[str, torch.Tensor],
) -> tuple[str, ...]:
    changed = []
    for selection in CHECKPOINT_SELECTIONS:
        score = float(getattr(validation, selection))
        if not math.isfinite(score):
            raise FloatingPointError(f"Non-finite validation {selection} loss")
        if selection in best and score >= best[selection]["selected_validation_loss"]:
            continue
        best[selection] = {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION, "checkpoint_kind": "selected",
            **run_metadata(config), "configuration": config,
            **config["training"]["checkpoint_selection"][selection],
            "selected_epoch": epoch, "selected_validation_loss": score,
            "validation_losses": asdict(validation), "model_state_dict": model_state,
        }
        changed.append(selection)
    return tuple(changed)


def validate_selected_checkpoint(
    checkpoint: Mapping[str, Any], config: Mapping[str, Any], *, selection: str,
    expected_epoch: int, expected_loss: float,
) -> None:
    validate_identity(checkpoint, config)
    expected = {"checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION, "checkpoint_kind": "selected",
                **config["training"]["checkpoint_selection"][selection],
                "selected_epoch": expected_epoch, "selected_validation_loss": expected_loss}
    for key, value in expected.items():
        if canonical_json(checkpoint.get(key)) != canonical_json(value):
            raise RuntimeError(f"Invalid {selection} checkpoint metadata: {key}")
    if (not 1 <= expected_epoch <= config["training"]["max_epochs"] or not math.isfinite(expected_loss)
            or checkpoint.get("validation_losses", {}).get(selection) != expected_loss):
        raise RuntimeError(f"Invalid {selection} checkpoint epoch/loss")
    validate_loss_values(checkpoint["validation_losses"], config, f"{selection} checkpoint losses")
    validate_model_state(checkpoint.get("model_state_dict"), config)


def make_resume_checkpoint(
    *, config: Mapping[str, Any], epoch: int, model_state: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer, shuffle_generator: torch.Generator,
    best: dict[str, dict[str, Any]], history: list[dict[str, Any]], elapsed_seconds: float,
) -> dict[str, Any]:
    return {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION, "checkpoint_kind": "latest",
        **run_metadata(config), "configuration": config, "epoch": epoch,
        "model_state_dict": model_state, "optimizer_state_dict": optimizer.state_dict(),
        "shuffle_generator_state": shuffle_generator.get_state(), "rng_state": capture_rng_state(),
        "best_checkpoints": best, **best_tracking_fields(best),
        "history": history, "elapsed_seconds": elapsed_seconds,
    }


def validate_resume_checkpoint(checkpoint: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    if (checkpoint.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION
            or checkpoint.get("checkpoint_kind") != "latest"):
        raise RuntimeError("Expected this experiment's latest checkpoint, not a baseline composed/component checkpoint")
    validate_identity(checkpoint, config)
    required = {"epoch", "model_state_dict", "optimizer_state_dict", "shuffle_generator_state",
                "rng_state", "best_checkpoints", "history", "elapsed_seconds"}
    if required.difference(checkpoint):
        raise RuntimeError(f"Incomplete resume checkpoint: {sorted(required.difference(checkpoint))}")
    epoch, history = checkpoint["epoch"], checkpoint["history"]
    if (not isinstance(history, list) or not 1 <= epoch <= config["training"]["max_epochs"] or len(history) != epoch
            or [row["epoch"] for row in history] != list(range(1, epoch + 1))):
        raise RuntimeError("Latest epoch/history mismatch")
    for row in history:
        for phase in ("train", "validation"):
            validate_loss_values(row.get(phase), config, f"{phase} history at epoch {row['epoch']}")
    validate_model_state(checkpoint["model_state_dict"], config)
    best = checkpoint["best_checkpoints"]
    if set(best) != set(CHECKPOINT_SELECTIONS):
        raise RuntimeError("Latest must contain both complete selected snapshots")
    for key, value in best_tracking_fields(best).items():
        if checkpoint.get(key) != value:
            raise RuntimeError(f"Latest best tracker mismatch: {key}")
    for selection in CHECKPOINT_SELECTIONS:
        row = min(history, key=lambda item: item["validation"][selection])
        validate_selected_checkpoint(best[selection], config, selection=selection,
                                     expected_epoch=row["epoch"], expected_loss=row["validation"][selection])
        if best[selection]["validation_losses"] != row["validation"]:
            raise RuntimeError(f"{selection} snapshot does not match its history epoch")


def save_status(path: Path, *, status: str, run_id: str, **extra: Any) -> None:
    atomic_write_json(path, {"status": status, "run_id": run_id, "updated_at": utc_now(), **extra})


def completed_result_evaluations(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    if (result.get("result_schema_version") != RESULT_SCHEMA_VERSION or result.get("status") != "completed"
            or set(result.get("evaluations", {})) != set(CHECKPOINT_SELECTIONS)):
        raise RuntimeError("Expected a completed run with both total and structure evaluations")
    config = result["configuration"]
    validate_identity(result, config)
    if result["training_result"]["epochs_completed"] != config["training"]["max_epochs"]:
        raise RuntimeError("Completed result has an unfinished training history")
    common = {key: value for key, value in result.items() if key != "evaluations"}
    records = []
    for selection in CHECKPOINT_SELECTIONS:
        evaluation = result["evaluations"][selection]
        expected = {**config["training"]["checkpoint_selection"][selection],
                    "selected_epoch": result["training_result"][f"best_{selection}_epoch"],
                    "selected_validation_loss": result["training_result"][f"best_{selection}_loss"]}
        for key, value in expected.items():
            if canonical_json(evaluation.get(key)) != canonical_json(value):
                raise RuntimeError(f"Invalid {selection} result metadata: {key}")
        if (not 1 <= evaluation["selected_epoch"] <= config["training"]["max_epochs"]
                or not math.isfinite(evaluation["selected_validation_loss"])
                or evaluation["validation_losses"][selection] != evaluation["selected_validation_loss"]
                or not evaluation.get("checkpoint_path")):
            raise RuntimeError(f"Invalid {selection} selected epoch/loss/path")
        metrics = evaluation["test_metrics"]
        validate_loss_values(evaluation["validation_losses"], config, f"{selection} result losses")
        count = config["observation"]["g05_count_per_sample"]
        optional_sign = set(GLOBAL_METRIC_NAMES[:4]) if count == 0 else set()
        for name in (*METRIC_NAMES, "observed_sample_fraction", "observations_per_sample"):
            value = metrics.get(name)
            if (name not in metrics or (name in optional_sign and value is not None)
                    or (name not in optional_sign and (value is None or not math.isfinite(value)))):
                raise RuntimeError(f"Incomplete or invalid {selection} test metrics: {name}")
        if metrics["observed_sample_fraction"] != float(count > 0) or metrics["observations_per_sample"] != count:
            raise RuntimeError(f"{selection} result observation counts do not match the configured G05 prefix")
        records.append({**common, **evaluation})
    return records


def repair_completed_run(
    result: Mapping[str, Any], config: Mapping[str, Any], paths: Mapping[str, Path],
    status_path: Path, result_path: Path, device: torch.device,
) -> None:
    validate_identity(result, config)
    completed_result_evaluations(result)
    latest = None
    for selection in CHECKPOINT_SELECTIONS:
        missing = not paths[selection].exists()
        if missing:
            if latest is None:
                if not paths["latest"].is_file():
                    raise RuntimeError("Cannot restore missing best checkpoint without latest.pt")
                latest = load_torch_checkpoint(paths["latest"], device)
                validate_resume_checkpoint(latest, config)
                if latest["epoch"] != config["training"]["max_epochs"]:
                    raise RuntimeError("Cannot restore a completed run from unfinished latest.pt")
            checkpoint = latest["best_checkpoints"][selection]
        else:
            checkpoint = load_torch_checkpoint(paths[selection], device)
        evaluation = result["evaluations"][selection]
        validate_selected_checkpoint(checkpoint, config, selection=selection,
                                     expected_epoch=evaluation["selected_epoch"],
                                     expected_loss=evaluation["selected_validation_loss"])
        if checkpoint["validation_losses"] != evaluation["validation_losses"]:
            raise RuntimeError("Selected checkpoint losses differ from completed result")
        if missing:
            atomic_torch_save(checkpoint, paths[selection])
    save_status(status_path, status="completed", run_id=run_id_for(config),
                **best_tracking_fields(result["evaluations"]), result_path=str(result_path.resolve()))


def train_and_evaluate_run(
    *, run_config: dict[str, Any], datasets: tuple[TensorDataset, TensorDataset, TensorDataset],
    experiment_results_dir: Path, experiment_checkpoint_dir: Path, device: torch.device,
) -> tuple[dict[str, Any], bool]:
    run_id = run_id_for(run_config)
    result_dir = experiment_results_dir / "runs" / run_id
    paths = run_checkpoint_paths(experiment_checkpoint_dir / run_id)
    result_path, status_path = result_dir / "result.json", result_dir / "status.json"
    config_path, history_path = result_dir / "config.json", result_dir / "history.json"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        repair_completed_run(result, run_config, paths, status_path, result_path, device)
        print(f"SKIP completed: {run_id}", flush=True)
        return result, True
    if config_path.exists():
        if canonical_json(json.loads(config_path.read_text(encoding="utf-8"))) != canonical_json(run_config):
            raise RuntimeError(f"Run configuration mismatch: {config_path}")
    else:
        atomic_write_json(config_path, run_config)
    settings = TrainingSettings(**{name: run_config["training"][name] for name in TrainingSettings.__dataclass_fields__})
    weights = LossWeights(**run_config["training"]["loss_weights"])
    stats = normalization_from_config(run_config)
    train, validation, test = datasets
    for name, dataset in zip(("train", "validation", "test"), datasets):
        if len(dataset) != run_config["split_counts"][name]:
            raise ValueError(f"{name} dataset count does not match the common split")
        mask = dataset.tensors[2]
        count = run_config["observation"]["g05_count_per_sample"]
        candidates = run_config["observation"]["candidate_count"]
        if (tuple(mask.shape) != (len(dataset), candidates, 1)
                or not torch.all(mask[:, :count] == 1) or not torch.all(mask[:, count:] == 0)):
            raise ValueError(f"{name} mask does not match the configured fixed G05 prefix")
    seed = run_config["training"]["seed"]
    set_reproducibility(seed)
    model = MODEL_REGISTRY[run_config["model"]["name"]].factory().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)
    train_loader = physics.create_data_loader(train, settings.batch_size, shuffle=True, seed=seed, device=device)
    validation_loader = physics.create_data_loader(validation, settings.batch_size, device=device)
    shuffle_generator = train_loader.generator
    assert shuffle_generator is not None
    best: dict[str, dict[str, Any]] = {}
    history: list[dict[str, Any]] = []
    start_epoch, elapsed_before = 1, 0.0
    resumed = paths["latest"].exists()
    if resumed:
        latest = load_torch_checkpoint(paths["latest"], device)
        validate_resume_checkpoint(latest, run_config)
        model.load_state_dict(latest["model_state_dict"], strict=True)
        # AdamW.load_state_dict casts moment buffers to each parameter's device;
        # non-capturable step counters retain the baseline's CPU placement.
        optimizer.load_state_dict(latest["optimizer_state_dict"])
        shuffle_generator.set_state(latest["shuffle_generator_state"])
        restore_rng_state(latest["rng_state"])
        best, history = dict(latest["best_checkpoints"]), list(latest["history"])
        start_epoch, elapsed_before = latest["epoch"] + 1, latest["elapsed_seconds"]
        for selection in CHECKPOINT_SELECTIONS:
            atomic_torch_save(best[selection], paths[selection])
        atomic_write_json(history_path, history)
    elif history_path.exists() or any(paths[selection].exists() for selection in CHECKPOINT_SELECTIONS):
        raise RuntimeError("Incomplete run has no latest.pt; preserve it and use a new experiment instead of restarting over its best files")
    save_status(status_path, status="running", run_id=run_id, resumed=resumed,
                next_epoch=start_epoch, **best_tracking_fields(best))
    started = time.perf_counter()
    print(f"RUN {run_id} | {device} | start epoch={start_epoch}", flush=True)
    for epoch in range(start_epoch, settings.max_epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, weights)
        validation_loss = run_epoch(model, validation_loader, weights=weights)
        history.append({"epoch": epoch, "train": asdict(train_loss), "validation": asdict(validation_loss)})
        state = copy_model_state(model)
        changed = update_best_checkpoints(best, config=run_config, epoch=epoch,
                                          validation=validation_loss, model_state=state)
        # One atomic authority holds training state AND both whole-epoch bests.
        latest = make_resume_checkpoint(config=run_config, epoch=epoch, model_state=state,
                                        optimizer=optimizer, shuffle_generator=shuffle_generator,
                                        best=best, history=history,
                                        elapsed_seconds=elapsed_before + time.perf_counter() - started)
        atomic_torch_save(latest, paths["latest"])
        for selection in changed:
            atomic_torch_save(best[selection], paths[selection])
        atomic_write_json(history_path, history)
        save_status(status_path, status="running", run_id=run_id, epoch=epoch, **best_tracking_fields(best))
        print(f"  epoch={epoch:03d}/{settings.max_epochs} train={train_loss.total:.6f} "
              f"val_total={validation_loss.total:.6f} val_structure={validation_loss.structure:.6f} "
              f"best_total={best['total']['selected_epoch']} best_structure={best['structure']['selected_epoch']}", flush=True)
    save_status(status_path, status="evaluating", run_id=run_id, **best_tracking_fields(best))
    evaluations = {}
    for selection in CHECKPOINT_SELECTIONS:
        selected = load_torch_checkpoint(paths[selection], device)
        validate_selected_checkpoint(selected, run_config, selection=selection,
                                     expected_epoch=best[selection]["selected_epoch"],
                                     expected_loss=best[selection]["selected_validation_loss"])
        model.load_state_dict(selected["model_state_dict"], strict=True)
        metrics = evaluate_model(model, test, stats, batch_size=settings.batch_size, weights=weights)
        evaluations[selection] = {
            **run_config["training"]["checkpoint_selection"][selection],
            "selected_epoch": selected["selected_epoch"],
            "selected_validation_loss": selected["selected_validation_loss"],
            "validation_losses": selected["validation_losses"],
            "checkpoint_path": str(paths[selection].resolve()), "test_metrics": metrics,
        }
        print(f"  TEST {selection}: position_mae={metrics['mean_position_mae']:.6f} "
              f"position_3d={metrics['mean_position_3d_error']:.6f} magnitude={metrics['charge_magnitude_mae']:.6f}", flush=True)
    result = {
        "result_schema_version": RESULT_SCHEMA_VERSION, **run_metadata(run_config),
        "configuration": run_config, "status": "completed", "completed_at": utc_now(),
        "training_result": {"epochs_completed": settings.max_epochs, **best_tracking_fields(best),
                            "resumed": resumed, "elapsed_seconds": elapsed_before + time.perf_counter() - started},
        "evaluations": evaluations,
        "artifacts": {"history": str(history_path.resolve()), "config": str(config_path.resolve()),
                      **{name: str(path.resolve()) for name, path in paths.items()}},
    }
    completed_result_evaluations(result)
    atomic_write_json(result_path, result)
    save_status(status_path, status="completed", run_id=run_id, **best_tracking_fields(best),
                result_path=str(result_path.resolve()))
    return result, False


def load_trained_model(path: Path, device: torch.device = torch.device("cpu")) -> tuple[RoutedChargeNet, physics.NormalizationStats, dict[str, Any]]:
    checkpoint = load_torch_checkpoint(path, device)
    if (checkpoint.get("protocol_version") != PROTOCOL_VERSION
            or checkpoint.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION
            or not isinstance(checkpoint.get("configuration"), Mapping)):
        raise RuntimeError("This is not a checkpoint from the five-charge routing experiment; baseline checkpoints stay separate")
    config = checkpoint["configuration"]
    selection = checkpoint.get("checkpoint_selection")
    if selection not in CHECKPOINT_SELECTIONS:
        raise RuntimeError("Use best_structure.pt or best_total.pt for inference")
    validate_selected_checkpoint(checkpoint, config, selection=selection,
                                 expected_epoch=checkpoint["selected_epoch"],
                                 expected_loss=checkpoint["selected_validation_loss"])
    model = MODEL_REGISTRY[config["model"]["name"]].factory().to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, normalization_from_config(config), checkpoint


def load_evaluation_data(
    result_root: Path, data_path: Path | None,
) -> tuple[dict[str, Any], physics.DatasetArrays, physics.DataSplit, Path]:
    """Reuse the training protocol; do not refit normalization or recreate a split."""
    protocol_path = result_root / "protocol.json"
    if not protocol_path.is_file():
        raise FileNotFoundError(f"Evaluation requires an existing protocol.json: {protocol_path}; no training will run")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    fingerprint = protocol.pop("protocol_fingerprint", None)
    if object_fingerprint(protocol) != fingerprint:
        raise RuntimeError("Saved evaluation protocol fingerprint is invalid")
    if (protocol.get("protocol_version") != PROTOCOL_VERSION
            or protocol.get("baseline_protocol_version") != physics.PROTOCOL_VERSION):
        raise RuntimeError("Saved protocol is not compatible with this five-charge routing experiment")
    if protocol["source_sha256"]["NewLearning9.py"] != file_sha256(Path(physics.__file__)):
        raise RuntimeError("NewLearning9.py changed since training; restore the saved physics/evaluation implementation")
    # ModelExperiment9.py may have gained evaluation-only support since training.
    # Preserve its old hash in the training identity and record current sources separately.
    data_path = (data_path if data_path is not None else Path(protocol["data"]["path"])).resolve()
    if not data_path.is_file():
        raise FileNotFoundError(f"Evaluation dataset not found: {data_path}; use --data for an identical relocated copy")
    if file_sha256(data_path) != protocol["data"]["sha256"]:
        raise RuntimeError("Evaluation dataset SHA256 differs from the training dataset")
    arrays = physics.load_dataset(data_path)
    if (len(arrays.target) != protocol["data"]["sample_count"]
            or any(list(getattr(arrays, name).shape) != protocol["data"][f"{name}_shape"]
                   for name in ("g00", "g05", "target"))):
        raise RuntimeError("Evaluation dataset shapes differ from the saved protocol")
    indices = {name: np.asarray(protocol["physics"]["split_indices"][name])
               for name in ("train", "validation", "test")}
    if any(value.ndim != 1 or value.size == 0 or value.dtype.kind not in "iu" for value in indices.values()):
        raise RuntimeError("Invalid saved split indices")
    if not np.array_equal(np.sort(np.concatenate(list(indices.values()))), np.arange(len(arrays.target))):
        raise RuntimeError("Saved splits must be disjoint and cover the training dataset exactly")
    split_path = result_root / "split_indices.npz"
    if split_path.exists():
        with np.load(split_path, allow_pickle=False) as saved_split:
            if any(name not in saved_split or not np.array_equal(saved_split[name], value)
                   for name, value in indices.items()):
                raise RuntimeError("Saved split does not match the evaluation protocol")
    normalization_path = result_root / "normalization.json"
    if (canonical_json(protocol["normalization"]) != canonical_json(protocol["physics"]["normalization"])
            or (normalization_path.exists() and canonical_json(json.loads(normalization_path.read_text(encoding="utf-8")))
                != canonical_json(protocol["normalization"]))):
        raise RuntimeError("Saved normalization does not match the evaluation protocol")
    return protocol, arrays, physics.DataSplit(**indices), data_path


@torch.no_grad()
def evaluate_only_run(
    *, run_config: dict[str, Any], test: TensorDataset, experiment_results_dir: Path,
    experiment_checkpoint_dir: Path, device: torch.device, batch_size: int | None = None,
) -> dict[str, Any]:
    """Evaluate completed training snapshots without an optimizer or any source writes."""
    run_id = run_id_for(run_config)
    paths = run_checkpoint_paths(experiment_checkpoint_dir / run_id)
    result_path = experiment_results_dir / "runs" / run_id / "result.json"
    if paths["latest"].is_file():
        latest = load_torch_checkpoint(paths["latest"], device)
        validate_resume_checkpoint(latest, run_config)
        if latest["epoch"] != run_config["training"]["max_epochs"]:
            raise RuntimeError(f"Cannot evaluate unfinished training: {run_id} (epoch {latest['epoch']}); no training will run")
        best = latest["best_checkpoints"]
        training_result = {"epochs_completed": latest["epoch"], "elapsed_seconds": latest["elapsed_seconds"],
                           **best_tracking_fields(best)}
    elif result_path.is_file():
        # Selected model files suffice when a completed result proves training finished.
        original_result = json.loads(result_path.read_text(encoding="utf-8"))
        validate_identity(original_result, run_config)
        completed_result_evaluations(original_result)
        best, training_result = original_result["evaluations"], original_result["training_result"]
    else:
        raise FileNotFoundError(f"No completed training checkpoint/result for {run_id}: {paths['latest']}; no training will run")
    if len(test) != run_config["split_counts"]["test"]:
        raise ValueError("Evaluation test dataset count does not match the saved split")
    selected_checkpoints = {}
    for selection in CHECKPOINT_SELECTIONS:
        path = paths[selection]
        if path.is_file():
            selected = load_torch_checkpoint(path, device)
        elif "model_state_dict" in best[selection]:
            # Recover in memory only; never repair or rewrite training artifacts.
            selected, path = best[selection], paths["latest"]
        else:
            raise FileNotFoundError(f"Missing evaluation checkpoint: {path}; no training will run")
        validate_selected_checkpoint(selected, run_config, selection=selection,
                                     expected_epoch=best[selection]["selected_epoch"],
                                     expected_loss=best[selection]["selected_validation_loss"])
        if selected["validation_losses"] != best[selection]["validation_losses"]:
            raise RuntimeError(f"{selection} checkpoint losses differ from the saved training result")
        selected_checkpoints[selection] = selected, path
    set_reproducibility(run_config["training"]["seed"])
    model = MODEL_REGISTRY[run_config["model"]["name"]].factory().to(device)
    model.eval()
    stats = normalization_from_config(run_config)
    weights = LossWeights(**run_config["training"]["loss_weights"])
    effective_batch_size = batch_size if batch_size is not None else run_config["training"]["batch_size"]
    evaluations = {}
    print(f"EVALUATE {run_id} | {device} | batch_size={effective_batch_size}", flush=True)
    for selection, (selected, path) in selected_checkpoints.items():
        model.load_state_dict(selected["model_state_dict"], strict=True)
        metrics = evaluate_model(model, test, stats, batch_size=effective_batch_size, weights=weights)
        evaluations[selection] = {
            **run_config["training"]["checkpoint_selection"][selection],
            "selected_epoch": selected["selected_epoch"], "selected_validation_loss": selected["selected_validation_loss"],
            "validation_losses": selected["validation_losses"], "checkpoint_path": str(path.resolve()),
            "checkpoint_source": f"best_checkpoints.{selection}" if path == paths["latest"] else "model_state_dict",
            "test_metrics": metrics,
        }
        print(f"  TEST {selection}: position_mae={metrics['mean_position_mae']:.6f} "
              f"position_3d={metrics['mean_position_3d_error']:.6f} magnitude={metrics['charge_magnitude_mae']:.6f}", flush=True)
    result = {
        "result_schema_version": RESULT_SCHEMA_VERSION, **run_metadata(run_config),
        "configuration": run_config, "status": "completed", "completed_at": utc_now(),
        "evaluation_only": True, "evaluation_batch_size": effective_batch_size,
        "training_result": training_result, "evaluations": evaluations,
    }
    completed_result_evaluations(result)
    return result


def result_to_row(record: Mapping[str, Any]) -> dict[str, Any]:
    config = record["configuration"]
    return {
        **run_metadata(config), "model": config["model"]["name"],
        "checkpoint_selection": record["checkpoint_selection"],
        "selection_objective": record["selection_objective"],
        "selected_epoch": record["selected_epoch"], "selected_validation_loss": record["selected_validation_loss"],
        **{f"selected_validation_{key}": value for key, value in record["validation_losses"].items()},
        "global_sign_in_selection_objective": record["global_sign_in_selection_objective"],
        "global_sign_metrics_note": record["global_sign_metrics_note"],
        "primary_metrics": canonical_json(record["primary_metrics"]),
        "parameter_count": config["model"]["parameter_count"]["total"],
        "epochs_completed": record["training_result"]["epochs_completed"],
        "elapsed_seconds": record["training_result"]["elapsed_seconds"],
        "checkpoint_path": record["checkpoint_path"], **record["test_metrics"],
    }


def index_results(records: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, str, float, int], Mapping[str, Any]]:
    indexed = {}
    for record in records:
        key = (record["protocol_fingerprint"], record["checkpoint_selection"], record["model_name"],
               record["g05_fraction"], record["seed"])
        if key in indexed:
            raise RuntimeError(f"Duplicate run/selection in reports: {key}")
        indexed[key] = record
    return indexed


def build_summary_rows(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    indexed = index_results(records)
    groups = sorted({key[:4] for key in indexed})
    summaries = []
    for group in groups:
        members = sorted((record for key, record in indexed.items() if key[:4] == group), key=lambda r: r["seed"])
        first = members[0]
        row = {
            "protocol_fingerprint": group[0], "checkpoint_selection": group[1],
            "model": group[2], "g05_fraction": group[3], "g05_count_per_sample": first["g05_count_per_sample"],
            "run_count": len(members), "seeds": ",".join(str(record["seed"]) for record in members),
            "selected_epochs_by_seed": canonical_json({str(r["seed"]): r["selected_epoch"] for r in members}),
            "selected_validation_losses_by_seed": canonical_json({str(r["seed"]): r["selected_validation_loss"] for r in members}),
            "run_fingerprints_by_seed": canonical_json({str(r["seed"]): r["run_fingerprint"] for r in members}),
            "global_sign_in_selection_objective": first["global_sign_in_selection_objective"],
            "global_sign_metrics_note": first["global_sign_metrics_note"],
        }
        for metric in METRIC_NAMES:
            values = [float(r["test_metrics"][metric]) for r in members if r["test_metrics"][metric] is not None]
            row.update({f"{metric}_count": len(values),
                        f"{metric}_mean": float(np.mean(values)) if values else None,
                        f"{metric}_std": sample_std(values)})
        summaries.append(row)
    return summaries


def build_pairwise_rows(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    indexed = index_results(records)
    keys = sorted({(key[0], key[1], key[3], key[4]) for key in indexed})
    rows = []
    model_a, model_b = DEFAULT_MODELS
    for protocol, selection, fraction, seed in keys:
        a = indexed.get((protocol, selection, model_a, fraction, seed))
        b = indexed.get((protocol, selection, model_b, fraction, seed))
        if a is None or b is None:
            continue
        for metric in METRIC_NAMES:
            value_a, value_b = a["test_metrics"][metric], b["test_metrics"][metric]
            delta = float(value_b - value_a) if value_a is not None and value_b is not None else None
            improvement = None if delta is None else (-delta if metric in LOWER_IS_BETTER else delta)
            rows.append({
                "protocol_fingerprint": protocol, "checkpoint_selection": selection,
                "selection_objective": a["selection_objective"],
                "model_a": model_a, "model_b": model_b, "g05_fraction": fraction,
                "g05_count_per_sample": a["g05_count_per_sample"], "metric": metric, "seed": seed,
                "metric_role": "primary" if metric in a["primary_metrics"] else "secondary",
                "value_a": value_a, "value_b": value_b, "delta_b_minus_a": delta,
                "improvement_b_over_a": improvement,
                "selected_epoch_a": a["selected_epoch"], "selected_epoch_b": b["selected_epoch"],
                "selected_validation_loss_a": a["selected_validation_loss"],
                "selected_validation_loss_b": b["selected_validation_loss"],
                "run_fingerprint_a": a["run_fingerprint"], "run_fingerprint_b": b["run_fingerprint"],
                "global_sign_in_selection_objective": a["global_sign_in_selection_objective"],
                "global_sign_metrics_note": a["global_sign_metrics_note"],
                "improvement_definition": "positive means g05_full_reconstruction is better",
            })
    return rows


def build_pairwise_summary_rows(pairs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in pairs:
        key = tuple(row[name] for name in ("protocol_fingerprint", "checkpoint_selection", "model_a", "model_b", "g05_fraction", "metric"))
        grouped.setdefault(key, []).append(row)
    summaries = []
    for key in sorted(grouped):
        members = sorted(grouped[key], key=lambda row: row["seed"])
        if len({row["seed"] for row in members}) != len(members):
            raise RuntimeError("Duplicate paired seed in comparison summary")
        valid = [row for row in members if row["delta_b_minus_a"] is not None]
        deltas = [row["delta_b_minus_a"] for row in valid]
        improvements = [row["improvement_b_over_a"] for row in valid]
        first = members[0]
        summaries.append({
            **{name: first[name] for name in ("protocol_fingerprint", "checkpoint_selection", "selection_objective",
                                             "model_a", "model_b", "g05_fraction", "g05_count_per_sample", "metric", "metric_role",
                                             "global_sign_in_selection_objective", "global_sign_metrics_note", "improvement_definition")},
            "paired_seed_count": len(valid), "paired_seeds": ",".join(str(row["seed"]) for row in valid),
            "delta_mean": float(np.mean(deltas)) if deltas else None, "delta_std": sample_std(deltas),
            "improvement_mean": float(np.mean(improvements)) if improvements else None,
            "improvement_std": sample_std(improvements),
        })
    return summaries


def load_completed_results(root: Path, protocol_fingerprint: str) -> list[dict[str, Any]]:
    records = []
    for path in (root / "runs").glob("*/result.json"):
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            print(f"WARNING: reports unchanged; unreadable result {path}: {error}", flush=True)
            raise
        if result.get("status") == "completed" and result.get("protocol_fingerprint") == protocol_fingerprint:
            records.extend(completed_result_evaluations(result))
    return records


def refresh_reports(root: Path, protocol_fingerprint: str) -> bool:
    try:
        records = load_completed_results(root, protocol_fingerprint)
    except (OSError, json.JSONDecodeError):
        return False
    indexed = index_results(records)
    runs = [result_to_row(indexed[key]) for key in sorted(indexed)]
    pairs = build_pairwise_rows(records)
    # Validate/build every table before publishing. Empty lists clear stale CSV rows.
    reports = (("runs.csv", runs), ("summary.csv", build_summary_rows(records)),
               ("pairwise_comparisons.csv", pairs), ("pairwise_summary.csv", build_pairwise_summary_rows(pairs)))
    for name, rows in reports:
        atomic_write_csv(root / name, rows)
    return True


def assert_outputs_close(first: physics.ModelOutput, second: physics.ModelOutput, *, exact: bool = True) -> None:
    for name in OUTPUT_FIELDS:
        torch.testing.assert_close(getattr(first, name), getattr(second, name),
                                   rtol=0 if exact else 1e-5, atol=0 if exact else 1e-6)


def assert_finite_output(output: physics.ModelOutput, batch_size: int) -> None:
    expected = ((batch_size, 5, 3), (batch_size, 5), (batch_size, 5), (batch_size,))
    for name, shape in zip(OUTPUT_FIELDS, expected):
        tensor = getattr(output, name)
        if tuple(tensor.shape) != shape or not torch.isfinite(tensor).all():
            raise AssertionError(f"Invalid five-charge output: {name}")


def assert_no_gradient(model: nn.Module, prefixes: tuple[str, ...]) -> None:
    if any(parameter.grad is not None and torch.any(parameter.grad != 0)
           for name, parameter in model.named_parameters() if name.startswith(prefixes)):
        raise AssertionError(f"Unexpected gradient in {prefixes}")


def batch_to_epoch_loss(loss: physics.BatchLoss) -> EpochLoss:
    return EpochLoss(**{name: None if (value := getattr(loss, name)) is None else float(value.detach())
                        for name in EpochLoss.__dataclass_fields__})


def run_checkpoint_smoke_test(
    model: RoutedChargeNet, optimizer: torch.optim.Optimizer, config: dict[str, Any],
    batch: tuple[torch.Tensor, ...], weights: LossWeights, device: torch.device,
) -> None:
    with torch.no_grad():
        reference = model(*batch[:3])
        validation = batch_to_epoch_loss(physics.calculate_losses(reference, *batch[3:], batch[2], weights))
    best: dict[str, dict[str, Any]] = {}
    state = copy_model_state(model)
    update_best_checkpoints(best, config=config, epoch=1, validation=validation, model_state=state)
    history = [{"epoch": 1, "train": asdict(validation), "validation": asdict(validation)}]
    shuffle = torch.Generator().manual_seed(config["training"]["seed"])
    latest = make_resume_checkpoint(config=config, epoch=1, model_state=state, optimizer=optimizer,
                                    shuffle_generator=shuffle, best=best, history=history, elapsed_seconds=0.0)
    with tempfile.TemporaryDirectory(prefix="m9-smoke-") as directory:
        paths = run_checkpoint_paths(Path(directory))
        atomic_torch_save(latest, paths["latest"])
        loaded = load_torch_checkpoint(paths["latest"], device)
        validate_resume_checkpoint(loaded, config)
        torch.testing.assert_close(loaded["shuffle_generator_state"], shuffle.get_state(), rtol=0, atol=0)
        for selection in CHECKPOINT_SELECTIONS:
            atomic_torch_save(best[selection], paths[selection])
            restored, stats, checkpoint = load_trained_model(paths[selection], device)
            if canonical_json(stats.to_dict()) != canonical_json(config["normalization"]):
                raise AssertionError("Checkpoint normalization changed")
            with torch.no_grad():
                assert_outputs_close(reference, restored(*batch[:3]))
            if checkpoint["selected_epoch"] != 1:
                raise AssertionError("Checkpoint epoch changed")


def run_smoke_tests(
    *, arrays: physics.DatasetArrays, split: physics.DataSplit, stats: physics.NormalizationStats,
    protocol: Mapping[str, Any], fractions: Sequence[float], device: torch.device,
) -> None:
    """Use only a small training subset; never choose a checkpoint with test data."""
    saved_rng = capture_rng_state()
    settings = TrainingSettings(**{name: protocol["training"][name] for name in TrainingSettings.__dataclass_fields__})
    weights = LossWeights(**protocol["training"]["loss_weights"])
    seed = 1729
    try:
        # A positive fraction is mandatory even when the requested run is G05=0.
        for fraction in sorted(set(fractions) | {0.0, 0.75}):
            dataset = physics.prepare_dataset(arrays, split.train[:min(8, len(split.train))], stats, fraction)
            batch = tuple(tensor.to(device) for tensor in dataset.tensors)
            g00, g05, mask, position, charge = batch
            models = []
            for name in DEFAULT_MODELS:
                set_reproducibility(seed)
                models.append(MODEL_REGISTRY[name].factory().to(device))
            a, b = models
            if parameter_counts(a) != parameter_counts(b):
                raise AssertionError("Model capacities differ")
            if [name for name, _ in a.named_modules()] != [name for name, _ in b.named_modules()]:
                raise AssertionError("Module construction order differs")
            for name, tensor in a.state_dict().items():
                torch.testing.assert_close(tensor, b.state_dict()[name], rtol=0, atol=0)
            set_reproducibility(seed)
            baseline = physics.ChargeNet().to(device)
            for name, tensor in baseline.state_dict().items():
                torch.testing.assert_close(tensor, a.state_dict()[name], rtol=0, atol=0)
            with torch.no_grad():
                assert_outputs_close(a(g00, g05, mask), baseline(g00, g05, mask))
                assert_outputs_close(a(g00, g05, mask), b(g00, g05, mask))
                assert_outputs_close(a(g00, g05, torch.zeros_like(mask)), b(g00, g05, torch.zeros_like(mask)))
            # Explicitly compare independently seeded loaders, including a second epoch.
            loaders = [physics.create_data_loader(dataset, 3, shuffle=True, seed=seed, device=device) for _ in models]
            for _ in range(2):
                first_order, second_order = (torch.cat([item[3] for item in loader]) for loader in loaders)
                torch.testing.assert_close(first_order, second_order, rtol=0, atol=0)
            for model, name in zip(models, DEFAULT_MODELS):
                optimizer = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)
                output = model(g00, g05, mask)
                assert_finite_output(output, len(g00))
                losses = physics.calculate_losses(output, position, charge, mask, weights)
                if not all(value is None or math.isfinite(value) for value in asdict(batch_to_epoch_loss(losses)).values()):
                    raise AssertionError("Non-finite smoke loss")
                order = [4, 1, 3, 0, 2]
                permuted = physics.calculate_losses(output, position[:, order], charge[:, order], mask, weights)
                for field in EpochLoss.__dataclass_fields__:
                    if getattr(losses, field) is not None:
                        torch.testing.assert_close(getattr(losses, field), getattr(permuted, field), atol=1e-6, rtol=1e-6)
                permutations = physics.charge_permutations(device)
                if permutations.shape != (120, 5) or torch.unique(permutations, dim=0).shape[0] != 120:
                    raise AssertionError("Matching must enumerate all 120 permutations")
                costs = physics.matching_cost(output, position, charge, weights)
                chosen = physics.minimum_cost_assignment(costs)
                torch.testing.assert_close(chosen.sort(dim=1).values, torch.arange(5, device=device).expand(len(g00), -1))
                oracle = torch.stack([costs[:, torch.arange(5, device=device), list(p)].sum(dim=1)
                                      for p in itertools.permutations(range(5))], dim=1).min(dim=1).values
                actual = costs.gather(2, chosen[:, :, None]).sum(dim=1).squeeze(-1)
                torch.testing.assert_close(actual, oracle)
                altered = physics.ModelOutput(output.position, output.magnitude, output.relative_sign_logit,
                                               output.global_sign_logit + 100)
                torch.testing.assert_close(physics.matching_cost(altered, position, charge, weights), costs, rtol=0, atol=0)
                patterns = physics.relative_sign_patterns(device, output.relative_sign_logit.dtype)
                if patterns.shape != (16, 5):
                    raise AssertionError("Expected 16 relative-sign patterns")
                torch.testing.assert_close(physics.decode_relative_signs(output.relative_sign_logit).prod(dim=1),
                                           torch.ones(len(g00), device=device), rtol=0, atol=0)
                optimizer.zero_grad(set_to_none=True)
                losses.structure.backward()
                assert_no_gradient(model, ("global_sign_head.",))
                if name == "g05_sign_only" or fraction == 0:
                    assert_no_gradient(model, ("g05_encoder.", "structure_context."))
                else:
                    if not has_nonzero_gradient(model, ("structure_context.",)):
                        raise AssertionError("Full reconstruction has no G05 structure route")
                    # The zero final projection initially blocks upstream gradients.
                    # Activate it with one structure update on this disposable model.
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    observed_input = g05.detach().clone().requires_grad_(True)
                    new_loss = physics.calculate_losses(model(g00, observed_input, mask), position, charge, mask, weights)
                    new_loss.structure.backward()
                    if (not has_nonzero_gradient(model, ("g05_encoder.",))
                            or not has_nonzero_gradient(model, ("structure_context.0.",))
                            or observed_input.grad is None or not torch.any(observed_input.grad[:, :, 2] != 0)):
                        raise AssertionError("Activated full route did not reach G05 encoder/input")
                    assert_no_gradient(model, ("global_sign_head.",))
                optimizer.zero_grad(set_to_none=True)
                new_loss = physics.calculate_losses(model(g00, g05, mask), position, charge, mask, weights)
                if new_loss.global_sign is not None:
                    new_loss.global_sign.backward()
                    assert_no_gradient(model, STRUCTURE_PREFIXES)
                    if not has_nonzero_gradient(model, ("g05_encoder.", "global_sign_head.")):
                        raise AssertionError("Global-sign branch has no gradient")
                elif fraction != 0:
                    raise AssertionError("Observed global sign was dropped")
                optimizer.zero_grad(set_to_none=True)
                physics.calculate_losses(model(g00, g05, mask), position, charge, mask, weights).total.backward()
                if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                    raise AssertionError("Non-finite gradient")
                optimizer.step()
                if not all(torch.isfinite(p).all() for p in model.parameters()):
                    raise AssertionError("Non-finite parameters after optimizer step")
                with torch.no_grad():
                    current = model(g00, g05, mask)
                    reversed_output = model(g00, g05 * g05.new_tensor((1, 1, -1)), mask)
                    # The unchanged baseline's batched CPU GEMM can differ by a few
                    # ulps when the +/- halves exchange row positions. Preserve the
                    # exact algebraic symmetry without claiming bitwise float identity.
                    for field in OUTPUT_FIELDS[:-1]:
                        torch.testing.assert_close(getattr(current, field), getattr(reversed_output, field), rtol=1e-6, atol=1e-7)
                    torch.testing.assert_close(current.global_sign_logit, -reversed_output.global_sign_logit, rtol=1e-6, atol=1e-7)
                    torch.testing.assert_close(current.global_sign_logit, model(g00 + 10, g05, mask).global_sign_logit, rtol=0, atol=0)
                    hidden = g05.masked_fill(~mask.bool().expand_as(g05), float("nan"))
                    assert_outputs_close(current, model(g00, hidden, mask))
                    missing = model(g00, torch.full_like(g05, float("nan")), torch.zeros_like(mask))
                    torch.testing.assert_close(missing.global_sign_logit, torch.zeros_like(missing.global_sign_logit), rtol=0, atol=0)
                    # Zero-observation equality must also hold after the context learned.
                    counterpart = MODEL_REGISTRY["g05_sign_only"].factory().to(device)
                    counterpart.load_state_dict(model.state_dict(), strict=True)
                    assert_outputs_close(missing, counterpart(g00, g05, torch.zeros_like(mask)))
                config = run_configuration(protocol, model_name=name, fraction=fraction, seed=seed)
                run_checkpoint_smoke_test(model, optimizer, config, batch, weights, device)
            print(f"SMOKE PASS: fraction={fraction:g}, points={physics.g05_count_for_fraction(fraction, arrays.g05.shape[1])}, "
                  f"parameters={parameter_counts(a)['total']} | capacity/init/order/mask/matching/parity/gradients/symmetry/finite/step/checkpoint", flush=True)
    finally:
        restore_rng_state(saved_rng)


def parse_csv_values(value: str, converter: Callable[[str], Any]) -> tuple[Any, ...]:
    return tuple(converter(item.strip()) for item in value.split(",") if item.strip())


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    defaults = TrainingSettings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, help=f"Dataset (training default: {DEFAULT_DATA_PATH}; evaluation: saved path)")
    parser.add_argument("--models", type=lambda value: parse_csv_values(value, str), default=DEFAULT_MODELS)
    parser.add_argument("--fractions", type=lambda value: parse_csv_values(value, float), default=(0.75,))
    parser.add_argument("--seeds", type=lambda value: parse_csv_values(value, int), default=(42,))
    parser.add_argument("--epochs", type=int, default=defaults.max_epochs, help="Training only; evaluation uses saved settings")
    parser.add_argument("--batch-size", type=int, help=f"Batch size (training default: {defaults.batch_size}; evaluation: saved size)")
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate, help="Training only")
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay, help="Training only")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--results-root", "--results-dir", dest="results_root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--checkpoint-root", "--checkpoint-dir", dest="checkpoint_root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--smoke-only", action="store_true", help="Check training subset only; do not create experiment artifacts or run full training")
    modes.add_argument("--evaluate-only", "--eval-only", action="store_true",
                       help="Evaluate saved checkpoints into a separate evaluations folder; never train or run smoke tests")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args(argv)
    if not args.evaluate_only:
        if args.data is None:
            args.data = DEFAULT_DATA_PATH
        if args.batch_size is None:
            args.batch_size = defaults.batch_size
    if (not args.models or len(set(args.models)) != len(args.models)
            or set(args.models).difference(MODEL_REGISTRY)):
        parser.error(f"--models must be a unique subset of {DEFAULT_MODELS}")
    if (not args.fractions or len(set(args.fractions)) != len(args.fractions)
            or any(not math.isfinite(f) or not 0 <= f <= 1 for f in args.fractions)):
        parser.error("--fractions must contain unique finite values in [0,1]")
    if (not args.seeds or len(set(args.seeds)) != len(args.seeds)
            or any(not 0 <= seed < 2**32 for seed in args.seeds)):
        parser.error("--seeds must contain unique integers in [0,2**32)")
    if args.epochs < 1 or (args.batch_size is not None and args.batch_size < 1):
        parser.error("--epochs and --batch-size must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("--learning-rate must be finite and positive")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("--weight-decay must be finite and nonnegative")
    if (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", args.experiment_name)
            or args.experiment_name.endswith(".")):
        parser.error("--experiment-name must be a simple directory name (letters, digits, underscores, hyphens, dots)")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    args.fractions = tuple(sorted(args.fractions))
    return args


def run_evaluation_matrix(
    *, args: argparse.Namespace, result_root: Path, checkpoint_root: Path, device: torch.device,
) -> Path:
    protocol, arrays, split, data_path = load_evaluation_data(result_root, args.data)
    fingerprint = object_fingerprint(protocol)
    stats = normalization_from_config(protocol)
    evaluation_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}_{uuid.uuid4().hex[:8]}"
    output_root = result_root / "evaluations" / evaluation_id
    context = {
        "mode": "evaluate_only", "created_at": utc_now(), "protocol_fingerprint": fingerprint,
        "source_protocol_path": str((result_root / "protocol.json").resolve()),
        "checkpoint_root": str(checkpoint_root.resolve()), "data_path": str(data_path),
        "data_sha256": protocol["data"]["sha256"], "test_sample_count": len(split.test),
        "models": args.models, "fractions": args.fractions, "seeds": args.seeds,
        "batch_size": args.batch_size if args.batch_size is not None else protocol["training"]["batch_size"],
        "source_sha256": {"ModelExperiment9.py": file_sha256(Path(__file__)),
                          "NewLearning9.py": file_sha256(Path(physics.__file__))},
        "environment": runtime_environment(device),
    }
    atomic_write_json(output_root / "evaluation.json", context)
    print(f"Evaluation only | device={device} | saved test samples={len(split.test)}", flush=True)
    print("Using saved normalization, split, loss weights and training settings; no smoke tests or training.", flush=True)
    print(f"Evaluation output: {output_root}", flush=True)
    completed = failed = 0
    for fraction in args.fractions:
        test = physics.prepare_dataset(arrays, split.test, stats, fraction)
        for seed in args.seeds:
            for model_name in args.models:
                # Build the identity from the SAVED protocol, including its old source hashes.
                config = run_configuration(protocol, model_name=model_name, fraction=fraction, seed=seed)
                run_id = run_id_for(config)
                try:
                    result = evaluate_only_run(run_config=config, test=test, experiment_results_dir=result_root,
                                               experiment_checkpoint_dir=checkpoint_root, device=device, batch_size=args.batch_size)
                except (Exception, KeyboardInterrupt) as error:
                    save_status(output_root / "runs" / run_id / "status.json",
                                status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                                run_id=run_id, error=f"{type(error).__name__}: {error}")
                    if isinstance(error, KeyboardInterrupt) or not args.continue_on_error:
                        raise
                    failed += 1
                    print(f"EVALUATION FAILED {run_id}: {error}", flush=True)
                    continue
                result["evaluation_context"] = context
                atomic_write_json(output_root / "runs" / run_id / "result.json", result)
                if not refresh_reports(output_root, fingerprint):
                    raise RuntimeError(f"Cannot refresh evaluation reports: {output_root}")
                completed += 1
        del test
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print(f"Evaluation complete: evaluated={completed}, failed={failed} | {output_root}", flush=True)
    if failed:
        raise RuntimeError(f"{failed} evaluations failed; inspect {output_root}; no training was performed")
    return output_root


def run_experiment_matrix(
    *, args: argparse.Namespace, arrays: physics.DatasetArrays, split: physics.DataSplit,
    stats: physics.NormalizationStats, protocol: Mapping[str, Any], device: torch.device,
    result_root: Path, checkpoint_root: Path,
) -> None:
    fingerprint = initialize_experiment_artifacts(experiment_results_dir=result_root, experiment_checkpoint_dir=checkpoint_root,
                                                  protocol=protocol, split=split)
    refresh_reports(result_root, fingerprint)
    print(f"Results: {result_root}\nCheckpoints: {checkpoint_root}", flush=True)
    completed = skipped = failed = 0
    for fraction in args.fractions:
        datasets = tuple(physics.prepare_dataset(arrays, getattr(split, name), stats, fraction)
                         for name in ("train", "validation", "test"))
        for seed in args.seeds:
            for model_name in args.models:
                config = run_configuration(protocol, model_name=model_name, fraction=fraction, seed=seed)
                try:
                    _, was_skipped = train_and_evaluate_run(run_config=config, datasets=datasets,
                                                            experiment_results_dir=result_root,
                                                            experiment_checkpoint_dir=checkpoint_root, device=device)
                except (Exception, KeyboardInterrupt) as error:
                    save_status(result_root / "runs" / run_id_for(config) / "status.json",
                                status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                                run_id=run_id_for(config), error=f"{type(error).__name__}: {error}",
                                traceback=traceback.format_exc(), recovery="Rerun the identical command; latest.pt is authoritative")
                    refresh_reports(result_root, fingerprint)
                    if isinstance(error, KeyboardInterrupt) or not args.continue_on_error:
                        raise
                    failed += 1
                    print(f"FAILED {run_id_for(config)}: {error}", flush=True)
                    continue
                skipped += int(was_skipped)
                completed += int(not was_skipped)
                refresh_reports(result_root, fingerprint)
        del datasets
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print(f"Experiment complete: new={completed}, skipped={skipped}, failed={failed}", flush=True)
    print(f"Paired comparisons: {result_root / 'pairwise_comparisons.csv'}\nAggregate: {result_root / 'pairwise_summary.csv'}", flush=True)
    if failed:
        raise RuntimeError(f"{failed} runs failed; inspect their status.json files")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    device = physics.DEVICE if args.device == "auto" else torch.device(args.device)
    result_root = args.results_root.resolve() / args.experiment_name
    checkpoint_root = args.checkpoint_root.resolve() / args.experiment_name
    if args.evaluate_only:
        if not (result_root / "protocol.json").is_file():
            raise FileNotFoundError(f"Evaluation requires an existing protocol.json: {result_root}; no training will run")
        with experiment_locks(result_root, checkpoint_root):
            run_evaluation_matrix(args=args, result_root=result_root, checkpoint_root=checkpoint_root, device=device)
        return
    settings = TrainingSettings(args.epochs, args.batch_size, args.learning_rate, args.weight_decay)
    arrays = physics.load_dataset(args.data.resolve())
    split = physics.create_data_split(len(arrays.target))
    stats = physics.calculate_normalization_stats(arrays, split.train)
    protocol = build_protocol(data_path=args.data.resolve(), arrays=arrays, split=split, stats=stats, settings=settings, device=device)
    print(f"Five-charge routing experiment | device={device} | G00={arrays.g00.shape} G05={arrays.g05.shape}", flush=True)
    print("Unchanged 120-permutation matching / 16 relative patterns / train-only normalization; G05=1 means all candidates", flush=True)
    run_smoke_tests(arrays=arrays, split=split, stats=stats, protocol=protocol, fractions=args.fractions, device=device)
    if args.smoke_only:
        print("Smoke-only complete; no test-set evaluation or experiment artifacts.", flush=True)
        return
    with experiment_locks(result_root, checkpoint_root):
        run_experiment_matrix(args=args, arrays=arrays, split=split, stats=stats, protocol=protocol,
                              device=device, result_root=result_root, checkpoint_root=checkpoint_root)


if __name__ == "__main__":
    main()
