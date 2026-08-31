"""Audit saved fixed-setting runs without training or evaluating the test set again."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Codes"))
import ModelExperiment10 as legacy
import numpy as np
import torch

FRACTIONS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
SETTINGS = dict(max_epochs=150, batch_size=128, learning_rate=0.001, weight_decay=0.0001)
REGULARIZATION = dict(structure_dropout=0.2, early_stopping_patience=20,
                      early_stopping_min_delta=0.0, early_stopping_min_epochs=0)


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def require(condition: bool, description: str) -> None:
    if not condition:
        raise RuntimeError(description)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--require-fresh", action="store_true", help="Require no checkpoint resumption in any run")
    args = parser.parse_args()
    require(0 <= args.seed < 2**32, "Seed must be one uint32 integer")
    name = f"seed{args.seed}_g05_sweep_dropout020"
    results = ROOT / "Modelexperiment11/fraction_sweep_results" / name
    checkpoints = ROOT / "Modelexperiment11/fraction_sweep_checkpoints" / name
    launches = ROOT / "Modelexperiment11/fraction_sweep_launches" / name
    saved = read(results / "protocol.json")
    protocol = {k: v for k, v in saved.items() if k != "protocol_fingerprint"}
    require(legacy.object_fingerprint(protocol) == saved["protocol_fingerprint"], "Protocol hash mismatch")
    for key, value in SETTINGS.items():
        require(protocol["training"][key] == value, f"Changed training setting: {key}")
    require(protocol["training"]["optimizer"] == "AdamW", "Changed optimizer")
    require(protocol["training"]["regularization"] == REGULARIZATION, "Changed regularization")
    require(protocol["training"]["loss_weights"] == dict.fromkeys(
        ("position", "magnitude", "relative_sign", "global_sign"), 1.0), "Changed loss weights")
    require(protocol["training"]["structure_reuse"] is False, "Structure reuse must be disabled")
    for source, digest in protocol["source_sha256"].items():
        require(legacy.file_sha256(ROOT / "Codes" / source) == digest, f"Source changed: {source}")
        require(legacy.file_sha256(launches / "sources" / source) == digest, f"Snapshot changed: {source}")
    data_path = ROOT / "Models/charge_dataset_5charges_v9.npz"
    require(legacy.file_sha256(data_path) == protocol["data"]["sha256"], "Dataset hash mismatch")
    arrays = legacy.physics.load_dataset(data_path)
    split = legacy.physics.create_data_split(len(arrays.target))
    require(protocol["physics"]["split_seed"] == 42, "Changed split seed")
    with np.load(results / "split_indices.npz", allow_pickle=False) as stored_split:
        for phase in ("train", "validation", "test"):
            require(np.array_equal(stored_split[phase], getattr(split, phase)), f"Changed {phase} split")
            require(np.array_equal(stored_split[phase], protocol["physics"]["split_indices"][phase]),
                    f"Protocol differs from saved {phase} split")
    stats = legacy.physics.calculate_normalization_stats(arrays, split.train).to_dict()
    require(stats == protocol["normalization"] == read(results / "normalization.json"),
            "Normalization is not the unchanged train-only normalization")
    del arrays

    run_rows, checkpoint_rows, expected_ids, expected_evaluations = [], [], set(), {}
    for fraction, model_name in itertools.product(FRACTIONS, legacy.DEFAULT_MODELS):
        config = legacy.run_configuration(protocol, model_name=model_name, fraction=fraction, seed=args.seed)
        run_id = legacy.run_id_for(config)
        expected_ids.add(run_id)
        directory = results / "runs" / run_id
        record, history = read(directory / "result.json"), read(directory / "history.json")
        require(read(directory / "config.json") == config, f"Configuration changed: {run_id}")
        require(read(directory / "status.json")["status"] == "completed", f"Incomplete run: {run_id}")
        legacy.validate_identity(record, config)
        legacy.completed_result_evaluations(record)
        require([row["epoch"] for row in history] == list(range(1, len(history) + 1)), "Nonconsecutive epochs")
        if args.require_fresh:
            require(record["training_result"]["resumed"] is False, f"Unexpected checkpoint reuse: {run_id}")
        paths = legacy.run_checkpoint_paths(checkpoints / run_id)
        latest = legacy.load_torch_checkpoint(paths["latest"], torch.device("cpu"))
        legacy.validate_resume_checkpoint(latest, config)
        require(latest["history"] == history, f"History differs from latest checkpoint: {run_id}")
        require(latest["epoch"] == record["training_result"]["epochs_completed"] == len(history), "Epoch count mismatch")
        for kind, path in paths.items():
            require(Path(record["artifacts"][kind]).resolve() == path.resolve(), "Checkpoint path escapes this seed")
            checkpoint_rows.append(dict(run_id=run_id, seed=args.seed, model=model_name, fraction=fraction,
                                        kind=kind, path=path.relative_to(ROOT).as_posix(),
                                        sha256=legacy.file_sha256(path), bytes=path.stat().st_size))
        for selection in legacy.CHECKPOINT_SELECTIONS:
            evaluation = record["evaluations"][selection]
            expected_evaluations[(run_id, selection)] = evaluation
            best_row = min(history, key=lambda row: row["validation"][selection])
            require(evaluation["selected_epoch"] == best_row["epoch"], "Checkpoint not selected by first validation minimum")
            require(evaluation["validation_losses"] == best_row["validation"], "Mixed or altered epoch losses")
            checkpoint = legacy.load_torch_checkpoint(paths[selection], torch.device("cpu"))
            legacy.validate_selected_checkpoint(checkpoint, config, selection=selection,
                                                 expected_epoch=best_row["epoch"],
                                                 expected_loss=best_row["validation"][selection])
            require(Path(evaluation["checkpoint_path"]).resolve() == paths[selection].resolve(), "Evaluation path mismatch")
            require(all(torch.equal(tensor, latest["best_checkpoints"][selection]["model_state_dict"][key])
                        for key, tensor in checkpoint["model_state_dict"].items()), "Best snapshot differs from latest authority")
            require(all(value is None or math.isfinite(value) for value in evaluation["test_metrics"].values()),
                    "Nonfinite test metric")
        selected = record["evaluations"]["structure"]
        run_rows.append(dict(run_id=run_id, model=model_name, fraction=fraction,
                             g05_count=config["observation"]["g05_count_per_sample"],
                             **record["training_result"], selected_epoch=selected["selected_epoch"],
                             validation_structure=selected["selected_validation_loss"],
                             test_metrics=selected["test_metrics"]))

    require({p.name for p in (results / "runs").iterdir() if p.is_dir()} == expected_ids, "Missing or extra run conditions")
    require({p.name for p in checkpoints.iterdir() if p.is_dir()} == expected_ids, "Mixed checkpoint conditions")
    require(len(list(checkpoints.rglob("*.pt"))) == 36, "Expected exactly 36 checkpoint files")
    with (results / "runs.csv").open(encoding="utf-8-sig", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    require(len(csv_rows) == 24 and {int(row["seed"]) for row in csv_rows} == {args.seed}, "CSV contains missing or mixed seeds")
    require({(row["run_id"], row["checkpoint_selection"]) for row in csv_rows} ==
            set(itertools.product(expected_ids, legacy.CHECKPOINT_SELECTIONS)), "CSV selection matrix mismatch")
    for row in csv_rows:
        evaluation = expected_evaluations[(row["run_id"], row["checkpoint_selection"])]
        require(int(row["selected_epoch"]) == evaluation["selected_epoch"], "CSV selected epoch mismatch")
        require(float(row["selected_validation_loss"]) == evaluation["selected_validation_loss"], "CSV validation score mismatch")
        for metric, value in evaluation["test_metrics"].items():
            require(row[metric] == "" if value is None else float(row[metric]) == value,
                    f"CSV test metric mismatch: {row['run_id']}/{metric}")

    # Check console chronology without evaluating test data again.
    starts, finishes, tests = [], set(), []
    active, finished = None, False
    for log in sorted(launches.glob("*.log")):
        for line in log.read_text(encoding="utf-8-sig").splitlines():
            match = re.match(r"RUN (\S+) \| .* \| start epoch=(\d+)", line)
            if match:
                active, start_epoch = match.group(1), int(match.group(2))
                require(active in expected_ids, "A different seed or condition was launched")
                starts.append((active, start_epoch))
                finished = False
            elif "TRAINING FINISHED:" in line:
                require(active in expected_ids, "Unidentified training completion")
                finishes.add(active)
                finished = True
            elif line.strip().startswith("TEST "):
                require(finished, "Test evaluation preceded training completion")
                tests.append((active, line.strip().split()[1].rstrip(":")))
    require(finishes == expected_ids, "Missing training completion logs")
    require(set(tests) == set(itertools.product(expected_ids, legacy.CHECKPOINT_SELECTIONS)), "Missing test evaluation logs")
    if args.require_fresh:
        require(len(starts) == 12 and all(epoch == 1 for _, epoch in starts), "Run did not start once at epoch 1")
        require(len(tests) == 24, "Unexpected repeat test evaluations")

    preservation = None
    baseline_path = launches / "preservation_before.json"
    if baseline_path.exists():
        baseline = read(baseline_path)
        changed = [item["path"] for item in baseline["files"]
                   if not (ROOT / item["path"]).is_file()
                   or legacy.file_sha256(ROOT / item["path"]) != item["sha256"]]
        require(not changed, f"Pre-existing experiment files changed: {changed}")
        preservation = dict(checked_files=len(baseline["files"]), changed_files=changed)

    # Supplementary comparison key: retain all scientific settings; exclude host and absolute data path.
    portable = {key: value for key, value in protocol.items() if key != "environment"}
    portable["data"] = {key: value for key, value in protocol["data"].items() if key != "path"}
    portable["physics"] = {key: value for key, value in protocol["physics"].items() if key != "data_path"}
    portable["experiment_matrix"] = dict(fractions=list(FRACTIONS), models=list(legacy.DEFAULT_MODELS))
    audit = dict(verified_at=legacy.utc_now(), seed=args.seed, status="passed", require_fresh=args.require_fresh,
                 runs=len(run_rows), total_epochs=sum(row["epochs_completed"] for row in run_rows),
                 elapsed_seconds_sum=sum(row["elapsed_seconds"] for row in run_rows),
                 checkpoint_count=len(checkpoint_rows), checkpoint_bytes=sum(row["bytes"] for row in checkpoint_rows),
                 test_evaluations=len(tests), test_recomputed_by_audit=False,
                 protocol_fingerprint=saved["protocol_fingerprint"],
                 portable_scientific_protocol_sha256=legacy.object_fingerprint(portable),
                 portable_key_note="Supplementary grouping key only; original protocol is unchanged. Hardware/software environment must still be reported.",
                 source_sha256=protocol["source_sha256"], data_sha256=protocol["data"]["sha256"],
                 preservation=preservation, run_details=run_rows)
    legacy.atomic_write_json(results / "seed_audit.json", audit)
    legacy.atomic_write_json(results / "checkpoint_hashes.json", checkpoint_rows)
    lines = [f"# Seed {args.seed}: fixed dropout020 fraction experiment", "",
             f"Verified {len(run_rows)}/12 independent runs, {audit['total_epochs']} epochs and 36 checkpoints.",
             "", "Settings: AdamW, lr=0.001, weight decay=0.0001, structure dropout=0.2, batch=128, "
             "maximum epochs=150, dual-objective patience=20, min_delta=0, min_epochs=0.",
             "Fixed 80/10/10 split (seed 42), train-only normalization, unchanged loss weights and 120-permutation matching.",
             "", "Primary table uses best_structure.pt selected only by validation. best_total.pt remains a separate saved evaluation; "
             "no checkpoint or fraction was chosen using test scores. Test metrics use the historical 1,000-sample test split, not a new holdout.",
             "", "| Fraction | G05 points | Model | Epochs | Best structure epoch | Validation structure | Test 3D error | Test magnitude MAE |",
             "|---:|---:|---|---:|---:|---:|---:|---:|"]
    for row in run_rows:
        label = "sign-only" if row["model"] == "g05_sign_only" else "full"
        lines.append(f"| {row['fraction']:g} | {row['g05_count']} | {label} | {row['epochs_completed']} | "
                     f"{row['selected_epoch']} | {row['validation_structure']:.6f} | "
                     f"{row['test_metrics']['mean_position_3d_error']:.6f} | {row['test_metrics']['charge_magnitude_mae']:.6f} |")
    lines += ["", "Only this seed is included. One seed cannot establish variance or statistical significance. "
              "Identical hardware/software is required for strict reproducibility; cross-device bitwise equality is not guaranteed.",
              "", f"Existing-artifact preservation: {preservation['checked_files']} files unchanged." if preservation else "",
              "", "[All recorded metrics](runs.csv) | [Summary](summary.csv) | [Paired model comparisons](pairwise_summary.csv) | "
              "[Protocol](protocol.json) | [Audit](seed_audit.json) | [Checkpoint hashes](checkpoint_hashes.json)",
              "", "From the project root, fresh execution on another computer:", "", "```powershell",
              f".\\Modelexperiment11\\run_fixed_seed.ps1 -Seed {args.seed}", "```", "",
              "Use that computer's assigned seed. The launcher rejects existing output directories unless -Resume is explicit. "
              "It checks frozen source/data hashes and never reads another seed's results or weights.", ""]
    (results / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({key: value for key, value in audit.items() if key != "run_details"}, indent=2))


if __name__ == "__main__":
    main()
