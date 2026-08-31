"""실제 데이터의 train/validation만 쓰는 정규화 비교. 테스트셋은 만들지도 않는다.

이 파일은 ModelExperiment10의 운영 학습/결과 파일을 생성하지 않는다. 설정 선택용
pilot의 범위, 모든 검증 이력, 실제 소스/데이터 해시를 별도 폴더에 남긴다.
같은 파일이 이미 있으면 덮어쓰지 않고 실패하므로 재실행은 다른 --label을 쓴다.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "Codes"))
import ModelExperiment10 as experiment
import NewLearning9 as physics
import torch


VARIANTS = {
    "baseline": (0.0, 1e-4),
    "dropout010": (0.1, 1e-4),
    "dropout020": (0.2, 1e-4),
    "decay001": (0.0, 1e-3),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="screening_seed43")
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--seeds", default="43")
    parser.add_argument("--models", default="g05_full_reconstruction")
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    output = Path(__file__).resolve().parent / args.label
    if output.exists():
        raise FileExistsError(f"Pilot output already exists: {output}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arrays = physics.load_dataset(experiment.DEFAULT_DATA_PATH)
    split = physics.create_data_split(len(arrays.target))
    stats = physics.calculate_normalization_stats(arrays, split.train)
    # test 인덱스로는 prepare_dataset을 호출하지 않는다. 설정 선택에 테스트 정답이
    # 들어가지 않게 train/validation loader 두 개만 구성한다.
    train = physics.prepare_dataset(arrays, split.train, stats, args.fraction)
    validation = physics.prepare_dataset(arrays, split.validation, stats, args.fraction)
    context = {
        "purpose": "validation-only hyperparameter pilot; no test-set evaluation or production checkpoint",
        "created_at": experiment.utc_now(), "arguments": vars(args),
        "source_sha256": {name: experiment.file_sha256(PROJECT / "Codes" / name)
                          for name in ("ModelExperiment10.py", "NewLearning9.py")},
        "pilot_sha256": experiment.file_sha256(Path(__file__)),
        "data_sha256": experiment.file_sha256(experiment.DEFAULT_DATA_PATH),
        "split_indices": {"train": split.train, "validation": split.validation},
        "normalization": stats.to_dict(), "environment": experiment.runtime_environment(device),
    }
    experiment.atomic_write_json(output / "context.json", context)
    summaries = []
    for name in args.models.split(","):
        for seed in map(int, args.seeds.split(",")):
            for variant in args.variants.split(","):
                dropout, decay = VARIANTS[variant]
                controls = experiment.RegularizationSettings(structure_dropout=dropout, early_stopping_patience=20)
                tracker = experiment.DualObjectiveEarlyStopping(controls)
                experiment.set_reproducibility(seed)
                model = experiment.MODEL_REGISTRY[name].factory(structure_dropout=dropout).to(device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=decay)
                loader = physics.create_data_loader(train, 128, shuffle=True, seed=seed, device=device)
                val_loader = physics.create_data_loader(validation, 128, device=device)
                history = []
                started = time.perf_counter()
                run_name = f"{name}__seed{seed}__{variant}"
                print(f"PILOT {run_name} | train={len(train)} validation={len(validation)} | test unused", flush=True)
                for epoch in range(1, args.epochs + 1):
                    train_loss = experiment.run_epoch(model, loader, optimizer)
                    val_loss = experiment.run_epoch(model, val_loader)
                    history.append({"epoch": epoch, "train": asdict(train_loss), "validation": asdict(val_loss)})
                    stopped = tracker.update(epoch, val_loss)
                    if epoch == 1 or epoch % 10 == 0 or stopped or epoch == args.epochs:
                        print(f"  epoch={epoch} train_structure={train_loss.structure:.6f} "
                              f"val_structure={val_loss.structure:.6f} val_total={val_loss.total:.6f} "
                              f"patience_used={tracker.bad_epochs}", flush=True)
                    if stopped:
                        break
                summary = {
                    "run_name": run_name, "model": name, "seed": seed, "fraction": args.fraction,
                    "variant": variant, "structure_dropout": dropout, "weight_decay": decay,
                    "epochs_completed": len(history), "max_epochs": args.epochs,
                    "stop_reason": "early_stopping" if tracker.stopped and len(history) < args.epochs else "max_epochs",
                    "elapsed_seconds": time.perf_counter() - started,
                    "parameter_count": experiment.parameter_counts(model),
                    **{f"best_{objective}_{field}": value
                       for objective in experiment.CHECKPOINT_SELECTIONS
                       for field, value in (
                           ("epoch", min(history, key=lambda h: h["validation"][objective])["epoch"]),
                           ("loss", min(h["validation"][objective] for h in history)),
                       )},
                }
                experiment.atomic_write_json(output / f"{run_name}.json", {**summary, "history": history,
                                                                            "early_stopping": tracker.state_dict()})
                summaries.append(summary)
                experiment.atomic_write_json(output / "summary.json", summaries)
                print(json.dumps(summary, ensure_ascii=False), flush=True)
                del model, optimizer, loader, val_loader
                if device.type == "cuda":
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
