"""현재 PC의 저장 호환성을 검사한다. 기존 가중치·결과·프로토콜은 읽기만 한다.

쓰기 검사는 Results/Models 각각에 새로 만든 임시 하위 폴더에 한정한다. 해당
경로가 허용 루트 안에 있음을 확인한 뒤에만 사용하며, 종료 시 그 폴더만 정리한다.
이 스크립트는 v9 모델을 v10 학습 모델로 변환하거나 재학습하지 않는다.
"""

from __future__ import annotations

import ctypes
import json
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import winreg
from pathlib import Path

OUT = Path(__file__).resolve().parent
PROJECT = OUT.parents[1]
sys.path.insert(0, str(PROJECT / "Codes"))
import ModelExperiment9 as m9
import ModelExperiment10 as m10
import NewLearning9 as physics
import numpy as np
import torch


class MemoryStatus(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def physical_memory() -> dict:
    memory = MemoryStatus()
    memory.dwLength = ctypes.sizeof(memory)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
        raise OSError("Cannot read physical memory status")
    return {"total_bytes": memory.ullTotalPhys, "available_bytes": memory.ullAvailPhys}


def main() -> None:
    source_paths = [PROJECT / "Codes" / name for name in ("ModelExperiment9.py", "NewLearning9.py", "test_model_experiment9.py")]
    existing_checkpoints = sorted(m9.DEFAULT_CHECKPOINT_ROOT.glob("*/*/*.pt"))
    # protocol뿐 아니라 기존 history/result/집계 등 모든 결과 파일의 바이트와
    # 수정 시각도 비교한다. 실제 원본에는 읽기 외의 작업을 하지 않는다.
    original_results = sorted(path for path in m9.DEFAULT_RESULTS_ROOT.rglob("*") if path.is_file())
    original_paths = [*source_paths, *existing_checkpoints, *original_results, m9.DEFAULT_DATA_PATH]
    before = {str(path): {"sha256": m10.file_sha256(path), "mtime_ns": path.stat().st_mtime_ns}
              for path in original_paths}
    disk = shutil.disk_usage(PROJECT)
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\FileSystem") as key:
        long_paths = winreg.QueryValueEx(key, "LongPathsEnabled")[0]
    environment = {"os": platform.platform(), "python": sys.version, "torch": torch.__version__,
                   "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
                   "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
                   "ram": physical_memory(), "disk_total_bytes": disk.total, "disk_free_bytes": disk.free,
                   "LongPathsEnabled": long_paths, "project": str(PROJECT)}
    original_ast = m10.source_ast_sha256(Path(physics.__file__))
    verified_sources = []
    for ref in ("ac59a6c", "f524bcf"):
        raw = subprocess.run(["git", "show", f"{ref}:Codes/NewLearning9.py"], cwd=PROJECT,
                             check=True, capture_output=True).stdout
        # 원본 소스는 임시 파일에서 해시할 뿐, 실제 Codes 파일에 덮어쓰지 않는다.
        with tempfile.TemporaryDirectory(prefix="source-audit-", dir=OUT) as directory:
            owned = Path(directory).resolve()
            assert owned.is_relative_to(OUT.resolve())
            path = owned / "original.py"
            path.write_bytes(raw)
            verified_sources.append({"ref": ref, "lf_sha256": m10.file_sha256(path),
                                     "same_executable_ast": m10.source_ast_sha256(path) == original_ast})
            path.write_bytes(raw.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
            verified_sources[-1]["crlf_sha256"] = m10.file_sha256(path)
    assert all(row["same_executable_ast"] for row in verified_sources)
    root_resolution, source_checks, data_checks = [], [], []
    saved_protocols = {}
    for seed in (41, 42, 43):
        path = m9.DEFAULT_RESULTS_ROOT / f"5point_routing_v1_seed{seed}" / "protocol.json"
        protocol = json.loads(path.read_text(encoding="utf-8"))
        saved_protocols[seed] = protocol
        args = m9.parse_args(["--evaluate-only", "--experiment-name", "5point_routing_v1", "--seeds", str(seed)])
        try:
            roots = m9.resolve_evaluation_roots(args)
            root_resolution.append({"seed": seed, "ok": True, "roots": list(map(str, roots))})
        except (RuntimeError, FileNotFoundError) as error:
            root_resolution.append({"seed": seed, "ok": False, "error": str(error)})
        try:
            original_verification = m9.validate_evaluation_source(protocol)
        except RuntimeError as error:
            original_verification = {"error": str(error)}
        source_checks.append({"seed": seed, "v9": original_verification,
                              "v10_physics_source_checker": m10.validate_evaluation_source(protocol)})
        try:
            chosen = m10.resolve_evaluation_dataset(protocol, None)
            data_checks.append({"seed": seed, "ok": True, "path": str(chosen)})
        except (RuntimeError, FileNotFoundError) as error:
            data_checks.append({"seed": seed, "ok": False, "error": str(error)})
    checkpoint_checks = []
    for path in existing_checkpoints:
        checkpoint = m9.load_torch_checkpoint(path, torch.device("cpu"))
        config = checkpoint["configuration"]
        assert checkpoint["run_id"] == path.parent.name
        if path.name == "latest.pt":
            m9.validate_resume_checkpoint(checkpoint, config)
        else:
            m9.validate_selected_checkpoint(checkpoint, config, selection=checkpoint["checkpoint_selection"],
                                            expected_epoch=checkpoint["selected_epoch"],
                                            expected_loss=checkpoint["selected_validation_loss"])
        checkpoint_checks.append({"path": str(path), "bytes": path.stat().st_size,
                                  "seed": config["training"]["seed"], "valid": True,
                                  "cpu_model_state": all(v.device.type == "cpu" for v in checkpoint["model_state_dict"].values())})
    # 기존 v9 가중치는 v9 전용 로더로만 읽는다. 위 물리 코드 동일성 검증을 통과한
    # seed43 데이터의 학습 샘플 4개로 CPU/CUDA 추론이 유한한지만 확인한다.
    selected = next(path for path in existing_checkpoints
                    if path.name == "best_structure.pt" and "g05_full_reconstruction__g05_100pct" in path.parent.name)
    data = physics.load_dataset(m10.resolve_evaluation_dataset(saved_protocols[43], None))
    inference = []
    for device in (torch.device("cpu"), torch.device("cuda")):
        model, stats, checkpoint = m9.load_trained_model(selected, device)
        dataset = physics.prepare_dataset(data, np.asarray(saved_protocols[43]["physics"]["split_indices"]["train"][:4]), stats, 1.0)
        with torch.no_grad():
            output = model(*(tensor.to(device) for tensor in dataset.tensors[:3]))
        assert all(torch.isfinite(getattr(output, name)).all() for name in m10.OUTPUT_FIELDS)
        inference.append({"device": str(device), "samples": 4, "finite": True,
                          "selected_epoch": checkpoint["selected_epoch"]})
        del model
    # v10의 실제 I/O 함수로 원본 payload의 복사본을 저장한다. 생산용 학습 결과나
    # 새 모델로 등록하지 않으며 namespace/schema를 바꾸지도 않는다.
    payload = m9.load_torch_checkpoint(selected, torch.device("cpu"))
    write_checks = []
    for storage_root in (PROJECT / "Results", PROJECT / "Models"):
        with tempfile.TemporaryDirectory(prefix=".m10-storage-probe-", dir=storage_root) as directory:
            owned = Path(directory).resolve()
            assert owned.is_relative_to(storage_root.resolve())
            nested = owned / ("한국어 경로 " + "a" * 70) / ("두 번째 " + "b" * 70) / ("세 번째 " + "c" * 70)
            assert nested.resolve().is_relative_to(owned)
            target = nested / "checkpoint_copy.pt"
            started = time.perf_counter()
            for attempt in range(2):
                m10.atomic_torch_save(payload, target)
                restored = m10.load_torch_checkpoint(target, torch.device("cuda"))
                for name, tensor in payload["model_state_dict"].items():
                    torch.testing.assert_close(restored["model_state_dict"][name], tensor, rtol=0, atol=0)
                assert all(t.device.type == "cpu" for t in restored["model_state_dict"].values())
                m10.atomic_write_json(nested / "결과.json", {"attempt": attempt, "note": "temporary storage probe"})
                m10.atomic_write_csv(nested / "결과.csv", [{"attempt": attempt}])
                m10.atomic_save_npz(nested / "split.npz", train=np.arange(4))
            lock_exclusive = False
            with m10.experiment_locks(nested):
                try:
                    with m10.experiment_locks(nested):
                        raise AssertionError("Second writer unexpectedly acquired the lock")
                except RuntimeError:
                    lock_exclusive = True
            assert not list(nested.glob("*.tmp"))
            write_checks.append({"root": str(storage_root), "checkpoint_path_characters": len(str(target)),
                                 "unicode_path": True, "atomic_create_and_replace": True, "cpu_load_for_cuda_request": True,
                                 "state_round_trip_exact": True, "exclusive_lock": lock_exclusive,
                                 "seconds": time.perf_counter() - started, "temporary_directory_cleaned": None})
        write_checks[-1]["temporary_directory_cleaned"] = not owned.exists()
        assert write_checks[-1]["temporary_directory_cleaned"]
    after = {str(path): {"sha256": m10.file_sha256(path), "mtime_ns": path.stat().st_mtime_ns}
             for path in original_paths}
    assert before == after, "An original file changed during the read-only audit"
    result = {"created_at": m10.utc_now(), "environment": environment,
              "source_files_verified_from_git": verified_sources, "source_checks": source_checks,
              "original_m9_default_root_resolution": root_resolution, "dataset_checks": data_checks,
              "existing_checkpoint_count": len(checkpoint_checks), "checkpoints": checkpoint_checks,
              "existing_checkpoint_bytes": sum(row["bytes"] for row in checkpoint_checks),
              "inference": inference, "write_checks": write_checks,
              "original_files_unchanged": True, "original_file_count": len(before), "original_file_manifest": before,
              "v10_source_sha256": m10.file_sha256(Path(m10.__file__)),
              "note": "v9 checkpoints remain v9; no model migration, training or test-metric evaluation was performed"}
    m10.atomic_write_json(OUT / "storage_audit.json", result)
    print(json.dumps({key: result[key] for key in ("environment", "existing_checkpoint_count", "existing_checkpoint_bytes",
                                                  "dataset_checks", "write_checks", "original_files_unchanged")}, indent=2))


if __name__ == "__main__":
    main()
