"""Read-only audit of the legacy experiment; only writes under Modelexperiment11."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Codes"))
import ModelExperiment10 as legacy
import NewLearning9 as physics
import numpy as np
import torch


def array_digest(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(str(value.dtype).encode() + str(value.shape).encode() + value.tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-preservation", action="store_true")
    args = parser.parse_args()
    out = ROOT / "Modelexperiment11" / "audit"
    manifest_path = out / "legacy_manifest.json"
    if args.verify_preservation:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        changed = [name for name, expected in manifest.items()
                   if not (ROOT / name).is_file() or legacy.file_sha256(ROOT / name) != expected]
        result = {"checked_files": len(manifest), "changed_or_missing": changed}
        legacy.atomic_write_json(out / "preservation_check.json", result)
        print(json.dumps(result))
        if changed:
            raise RuntimeError("Legacy files were changed")
        return
    if (out / "existing_audit.json").exists():
        raise FileExistsError("Audit already exists; preserve it")
    if not manifest_path.exists():
        paths = [p for directory in ("Codes", "Documents", "Results", "Models")
                 for p in (ROOT / directory).rglob("*")
                 if p.is_file() and "__pycache__" not in p.parts]
        paths += [ROOT / name for name in ("requirements.txt", "charge_dataset.npz", "Models.dvc", ".gitignore")]
        legacy.atomic_write_json(manifest_path,
                                 {p.relative_to(ROOT).as_posix(): legacy.file_sha256(p) for p in paths})
    arrays = physics.load_dataset(physics.DEFAULT_DATA_PATH)
    split = physics.create_data_split(len(arrays.target))
    stats = physics.calculate_normalization_stats(arrays, split.train)
    with np.load(physics.DEFAULT_DATA_PATH, allow_pickle=False) as archive:
        archive_info = {key: {"shape": list(archive[key].shape), "dtype": str(archive[key].dtype),
                              "sha256": array_digest(archive[key]),
                              **({"value": archive[key].tolist()} if archive[key].size <= 32 else {})}
                        for key in archive.files}
    digests = [hashlib.sha256(row.tobytes()).hexdigest() for row in arrays.g00]
    sets = {name: {digests[i] for i in getattr(split, name)} for name in ("train", "validation", "test")}
    overlap = {f"{a}_{b}": len(sets[a] & sets[b])
               for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))}
    histories = []
    for path in sorted((ROOT / "Results" / "new_learning9_experiments").glob("*/runs/*/history.json")):
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not rows:
            continue
        best = min(rows, key=lambda row: row["validation"]["structure"])
        histories.append({"path": path.relative_to(ROOT).as_posix(), "epochs": len(rows),
                          "best_structure_epoch": best["epoch"],
                          "best_validation": best["validation"], "train_at_best": best["train"],
                          "last_validation": rows[-1]["validation"], "last_train": rows[-1]["train"]})
    pilots = []
    pilot_root = ROOT / "Results" / "model_experiment10_validation"
    for path in sorted(pilot_root.glob("*/g05*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        pilots.append({key: value for key, value in record.items() if key != "history"})
    contexts = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(pilot_root.glob("*/context.json"))]
    checkpoint_paths = sorted((ROOT / "Models" / "new_learning9_experiments").rglob("*.pt"))
    inspected = []
    for model_name in legacy.DEFAULT_MODELS:
        path = next(p for p in checkpoint_paths if p.name == "best_structure.pt" and model_name in p.parent.name)
        saved = legacy.load_torch_checkpoint(path, torch.device("cpu"))
        inspected.append({"path": path.relative_to(ROOT).as_posix(), "sha256": legacy.file_sha256(path),
                          "protocol_version": saved.get("protocol_version"),
                          "selected_epoch": saved.get("selected_epoch"),
                          "selected_validation_loss": saved.get("selected_validation_loss"),
                          "configuration": saved.get("configuration"),
                          "finite_model_state": all(torch.isfinite(v).all().item()
                                                    for v in saved["model_state_dict"].values())})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    legacy.set_reproducibility(41)
    torch.use_deterministic_algorithms(True)
    model = legacy.MODEL_REGISTRY["g05_full_reconstruction"].factory().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train = physics.prepare_dataset(arrays, split.train, stats, 1.0)
    validation = physics.prepare_dataset(arrays, split.validation, stats, 1.0)
    loader = physics.create_data_loader(train, 128, shuffle=True, seed=41, device=device)
    val_loader = physics.create_data_loader(validation, 128, device=device)
    benchmark = []
    for epoch in range(1, 4):
        started = time.perf_counter()
        train_loss = legacy.run_epoch(model, loader, optimizer)
        val_loss = legacy.run_epoch(model, val_loader)
        benchmark.append({"epoch": epoch, "seconds": time.perf_counter() - started,
                          "train_structure": train_loss.structure, "validation_structure": val_loss.structure})
        print("Benchmark:", benchmark[-1], flush=True)
    report = {"created_at": legacy.utc_now(), "data_path": str(physics.DEFAULT_DATA_PATH),
              "data_sha256": legacy.file_sha256(physics.DEFAULT_DATA_PATH), "archive": archive_info,
              "split_counts": {name: len(getattr(split, name)) for name in sets},
              "split_seed": physics.DATA_SPLIT_SEED, "duplicate_g00_rows": len(digests) - len(set(digests)),
              "cross_split_identical_g00": overlap, "normalization": stats.to_dict(),
              "legacy_histories": histories, "legacy_pilots": pilots,
              "pilot_data_hashes": sorted({c["data_sha256"] for c in contexts}),
              "pilot_normalization_matches_current": all(c["normalization"] == stats.to_dict() for c in contexts),
              "checkpoint_count": len(checkpoint_paths), "inspected_checkpoints": inspected,
              "v10_checkpoint_count": len(list((ROOT / "Models" / "new_learning10_experiments").rglob("*.pt"))),
              "environment": legacy.runtime_environment(device),
              "benchmark_strict_determinism": True, "benchmark": benchmark,
              "test_evaluated": False}
    legacy.atomic_write_json(out / "existing_audit.json", report)
    print(json.dumps({key: report[key] for key in ("data_sha256", "split_counts", "duplicate_g00_rows",
                     "cross_split_identical_g00", "pilot_normalization_matches_current", "checkpoint_count",
                     "v10_checkpoint_count", "benchmark")}, indent=2))


if __name__ == "__main__":
    main()
