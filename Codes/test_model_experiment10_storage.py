"""PC 간 경로 이동, Python AST 형식, Windows 실제 파일 I/O의 회귀 검증.

실제 학습 폴더는 건드리지 않는다. 모든 모델·데이터·프로토콜은 이 테스트가 만든
임시 폴더에만 저장하며, 작은 실제 optimizer 업데이트 후 저장된 모델을 평가한다.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import ModelExperiment10 as experiment
import NewLearning9 as physics
import generate_charge_dataset as generator
import numpy as np
import torch


class StorageCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, previous_threads)
        cls.temporary = tempfile.TemporaryDirectory(prefix="m10-storage-tests-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name).resolve()
        cls.data_path = cls.root / "five.npz"
        generator.save_dataset(generator.generate_dataset(sample_count=20, g05_point_count=32,
                                                          seed=91, charge_count=5), cls.data_path)
        cls.arrays = physics.load_dataset(cls.data_path)
        cls.split = physics.create_data_split(20)
        cls.stats = physics.calculate_normalization_stats(cls.arrays, cls.split.train)
        cls.datasets = tuple(physics.prepare_dataset(cls.arrays, getattr(cls.split, name), cls.stats, .75)
                             for name in ("train", "validation", "test"))

    def setUp(self) -> None:
        self.case_root = self.root / hashlib.sha256(self._testMethodName.encode()).hexdigest()[:10]
        self.case_root.mkdir()

    def protocol(self) -> dict:
        return experiment.build_protocol(data_path=self.data_path, arrays=self.arrays, split=self.split,
                                         stats=self.stats, settings=physics.TrainingSettings(max_epochs=1, batch_size=8),
                                         regularization=experiment.RegularizationSettings(structure_dropout=.25),
                                         device=torch.device("cpu"))

    def prepare(self, result_root: Path, checkpoint_root: Path, protocol: dict) -> None:
        experiment.initialize_experiment_artifacts(experiment_results_dir=result_root,
                                                  experiment_checkpoint_dir=checkpoint_root,
                                                  protocol=protocol, split=self.split)

    def train_one(self, results: Path, checkpoints: Path, protocol: dict, model: str = "g05_full_reconstruction") -> dict:
        config = experiment.run_configuration(protocol, model_name=model, fraction=.75, seed=42)
        with contextlib.redirect_stdout(io.StringIO()):
            result, _ = experiment.train_and_evaluate_run(run_config=config, datasets=self.datasets,
                                                         experiment_results_dir=results, experiment_checkpoint_dir=checkpoints,
                                                         device=torch.device("cpu"))
        return result

    def arguments(self, root: Path) -> list[str]:
        return ["--evaluate-only", "--device", "cpu", "--experiment-name", "trial", "--seeds", "42",
                "--results-root", str(root / "results"), "--checkpoint-root", str(root / "checkpoints")]

    def snapshot(self, *roots: Path) -> dict:
        return {path: (path.read_bytes(), path.stat().st_mtime_ns)
                for root in roots for path in root.rglob("*") if path.is_file()}

    def test_seed_results_with_common_or_seed_checkpoint_layout_and_exact_overrides(self) -> None:
        for layout in ("common", "seed"):
            with self.subTest(layout=layout):
                root = self.case_root / layout
                results = root / "results" / "trial_seed42"
                checkpoints = root / "checkpoints" / ("trial" if layout == "common" else "trial_seed42")
                protocol = self.protocol()
                # 초기 v10처럼 AST 정보가 없고, Windows CRLF 원본 SHA만 있는 경우.
                protocol.pop("source_ast_sha256")
                protocol.pop("source_ast_format")
                protocol["source_sha256"]["NewLearning9.py"] = "8cd11ae42ff69a6520b2840023c88d204382d7bb047d8281f824f163e25dfba4"
                self.prepare(results, checkpoints, protocol)
                originals = {model: self.train_one(results, checkpoints, protocol, model) for model in experiment.DEFAULT_MODELS}
                before = self.snapshot(results, checkpoints)
                args = self.arguments(root)
                with contextlib.ExitStack() as stack:
                    for module, name in ((experiment, "train_and_evaluate_run"), (experiment, "run_smoke_tests"),
                                         (experiment, "run_epoch"), (experiment, "atomic_torch_save"),
                                         (physics, "create_data_split"), (physics, "calculate_normalization_stats"),
                                         (torch.optim, "AdamW")):
                        stack.enter_context(mock.patch.object(module, name, side_effect=AssertionError(f"unexpected {name}")))
                    output = stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                    experiment.main(args)
                    experiment.main([*args, "--experiment-name", "unused", "--evaluation-results-dir", str(results),
                                     "--evaluation-checkpoint-dir", str(checkpoints), "--structure-dropout", "0.9"])
                self.assertIn("seed-specific results", output.getvalue())
                if layout == "seed":
                    self.assertIn("matching checkpoint directory", output.getvalue())
                self.assertIn("executable code matches training", output.getvalue())
                evaluations = list((results / "evaluations").glob("*/runs/*/result.json"))
                self.assertEqual(len(evaluations), 4)
                for path in evaluations:
                    value = json.loads(path.read_text(encoding="utf-8"))
                    original = originals[value["model_name"]]
                    self.assertEqual(value["configuration"], original["configuration"])
                    context = value["evaluation_context"]
                    self.assertEqual(context["checkpoint_root"], str(checkpoints.resolve()))
                    self.assertEqual(context["source_protocol_path"], str((results / "protocol.json").resolve()))
                    self.assertEqual(context["source_compatibility"]["verification"], "identical_executable_ast")
                    for selection in experiment.CHECKPOINT_SELECTIONS:
                        self.assertEqual(value["evaluations"][selection]["test_metrics"], original["evaluations"][selection]["test_metrics"])
                for path, (content, modified) in before.items():
                    self.assertEqual(path.read_bytes(), content)
                    self.assertEqual(path.stat().st_mtime_ns, modified)
                self.assertFalse((root / "results" / "trial").exists())

    def test_explicit_missing_paths_and_multiple_seeds_are_not_guessed(self) -> None:
        root = self.case_root
        results, checkpoints = root / "results" / "trial_seed42", root / "checkpoints" / "trial_seed42"
        protocol = self.protocol()
        self.prepare(results, checkpoints, protocol)
        self.train_one(results, checkpoints, protocol)
        args = self.arguments(root)
        with self.assertRaisesRegex(FileNotFoundError, "evaluation-results-dir"):
            experiment.resolve_evaluation_roots(experiment.parse_args([*args, "--seeds", "42,43"]))
        with self.assertRaisesRegex(FileNotFoundError, "evaluation-results-dir"):
            experiment.resolve_evaluation_roots(experiment.parse_args([*args, "--evaluation-results-dir", str(root / "absent")]))
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(FileNotFoundError, "evaluation-checkpoint-dir"):
            experiment.resolve_evaluation_roots(experiment.parse_args([*args, "--evaluation-checkpoint-dir", str(root / "absent")]))
        with contextlib.redirect_stderr(io.StringIO()):
            for flag in ("--evaluation-results-dir", "--evaluation-checkpoint-dir"):
                with self.assertRaises(SystemExit):
                    experiment.parse_args([flag, str(root)])

    def test_checkpoint_discovery_matches_ids_and_rejects_ambiguous_alternatives(self) -> None:
        root = self.case_root
        results, aligned = root / "results" / "moved_results", root / "checkpoints" / "moved_results"
        protocol = self.protocol()
        self.prepare(results, aligned, protocol)
        result = self.train_one(results, aligned, protocol)
        args = [*self.arguments(root), "--models", "g05_full_reconstruction", "--evaluation-results-dir", str(results)]
        # 이름이 비슷한 폴더라도 요청 seed/run ID가 다르면 선택할 수 없다.
        primary = root / "checkpoints" / "trial"
        wrong = experiment.run_configuration(protocol, model_name="g05_full_reconstruction", fraction=.75, seed=43)
        wrong_path = experiment.run_checkpoint_paths(primary / experiment.run_id_for(wrong))["latest"]
        experiment.atomic_torch_save({"resolver_test_only": True}, wrong_path)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(experiment.resolve_evaluation_roots(experiment.parse_args(args)), (results, aligned))
        other = root / "checkpoints" / "trial_seed42" / result["run_id"] / "best_structure.pt"
        other.parent.mkdir(parents=True)
        other.write_bytes((aligned / result["run_id"] / "best_structure.pt").read_bytes())
        with self.assertRaisesRegex(RuntimeError, "Ambiguous"):
            experiment.resolve_evaluation_roots(experiment.parse_args(args))
        # 명시 경로는 모호성을 해소한다. 공통 원래 폴더에 일치 파일이 있으면 우선한다.
        self.assertEqual(experiment.resolve_evaluation_roots(experiment.parse_args(
            [*args, "--evaluation-checkpoint-dir", str(aligned)])), (results, aligned))
        selected_primary = primary / result["run_id"] / "best_structure.pt"
        selected_primary.parent.mkdir(parents=True)
        selected_primary.write_bytes(other.read_bytes())
        self.assertEqual(experiment.resolve_evaluation_roots(experiment.parse_args(args)), (results, primary))

    def test_missing_saved_dataset_uses_identical_local_copy_without_protocol_rewrite(self) -> None:
        root = self.case_root
        protocol = self.protocol()
        protocol["data"]["path"] = str(root / "old_computer" / "five.npz")
        results, checkpoints = root / "results", root / "checkpoints"
        self.prepare(results, checkpoints, protocol)
        before = self.snapshot(results, checkpoints)
        with mock.patch.object(experiment, "DEFAULT_DATA_PATH", self.data_path), mock.patch.object(
            experiment, "PROJECT_DIR", root,
        ), mock.patch.object(physics, "create_data_split", side_effect=AssertionError("saved split only")), mock.patch.object(
            physics, "calculate_normalization_stats", side_effect=AssertionError("saved normalization only"),
        ), contextlib.redirect_stdout(io.StringIO()) as output:
            restored, arrays, split, path = experiment.load_evaluation_data(results, None)
        self.assertEqual(restored, protocol)
        self.assertEqual(path, self.data_path.resolve())
        np.testing.assert_array_equal(arrays.target, self.arrays.target)
        np.testing.assert_array_equal(split.test, self.split.test)
        self.assertEqual(self.snapshot(results, checkpoints), before)
        self.assertIn("byte-identical relocated dataset", output.getvalue())

    def test_dataset_resolution_never_substitutes_mismatched_or_explicit_missing_data(self) -> None:
        root = self.case_root
        protocol = self.protocol()
        protocol["data"]["path"] = str(root / "old_computer" / "five.npz")
        invalid = root / "different.npz"
        invalid.write_bytes(b"different data, do not read as NPZ")
        with mock.patch.object(experiment, "DEFAULT_DATA_PATH", invalid), mock.patch.object(experiment, "PROJECT_DIR", root):
            with self.assertRaisesRegex(RuntimeError, "SHA256 differs"):
                experiment.resolve_evaluation_dataset(protocol, None)
        with mock.patch.object(experiment, "DEFAULT_DATA_PATH", self.data_path), mock.patch.object(experiment, "PROJECT_DIR", root):
            with self.assertRaises(FileNotFoundError):
                experiment.resolve_evaluation_dataset(protocol, root / "explicit_missing.npz")
            with self.assertRaisesRegex(RuntimeError, "SHA256 differs"):
                experiment.resolve_evaluation_dataset(protocol, invalid)
            protocol["data"]["path"] = str(invalid)
            with self.assertRaisesRegex(RuntimeError, "SHA256 differs"):
                experiment.resolve_evaluation_dataset(protocol, None)

    def test_ast_format_legacy_newline_hashes_and_real_code_changes(self) -> None:
        # 현재 Python의 default AST와 달라도 검증된 LF/CRLF 원본은 같은 계산 코드다.
        expected = "38a0d1b746fa0275e1e1efa2b303b7c164179d406121863e8392a2ff08684263"
        self.assertEqual(experiment.source_ast_sha256(Path(physics.__file__)), expected)
        for original_hash in experiment.LEGACY_BASELINE_AST_SHA256:
            protocol = {"source_sha256": {"NewLearning9.py": original_hash}}
            result = experiment.validate_evaluation_source(protocol)
            self.assertIn(result["verification"], ("identical_file", "identical_executable_ast"))
        original, documented, changed = (self.case_root / name for name in ("original.py", "documented.py", "changed.py"))
        original.write_text('"""Old docs."""\nthreshold = 1.0\nclass Model:\n    def predict(self):\n        return threshold\n', encoding="utf-8")
        documented.write_text('# New comment\n"""새 설명."""\nthreshold=1.0\nclass Model:\n    def predict(self):\n        """예측 설명."""\n        return threshold\n', encoding="utf-8")
        changed.write_text(documented.read_text(encoding="utf-8").replace("threshold=1.0", "threshold=2.0"), encoding="utf-8")
        protocol = {"source_sha256": {"NewLearning9.py": experiment.file_sha256(original)},
                    "source_ast_sha256": {"NewLearning9.py": experiment.source_ast_sha256(original)},
                    "source_ast_format": experiment.SOURCE_AST_FORMAT}
        with mock.patch.object(physics, "__file__", str(documented)):
            self.assertEqual(experiment.validate_evaluation_source(protocol)["verification"], "identical_executable_ast")
            legacy = copy.deepcopy(protocol)
            legacy.pop("source_ast_format")
            legacy["source_ast_sha256"]["NewLearning9.py"] = experiment.source_ast_sha256(original, legacy_default=True)
            self.assertEqual(experiment.validate_evaluation_source(legacy)["verification"], "identical_executable_ast")
            unknown = {"source_sha256": {"NewLearning9.py": "unknown-source"}}
            with self.assertRaisesRegex(RuntimeError, "compatibility cannot be verified"):
                experiment.validate_evaluation_source(unknown)
            protocol["source_ast_format"] = "unknown-serializer"
            with self.assertRaisesRegex(RuntimeError, "Unknown"):
                experiment.validate_evaluation_source(protocol)
        protocol["source_ast_format"] = experiment.SOURCE_AST_FORMAT
        with mock.patch.object(physics, "__file__", str(changed)):
            with self.assertRaisesRegex(RuntimeError, "executable code differs"):
                experiment.validate_evaluation_source(protocol)

    @unittest.skipUnless(os.name == "nt", "Windows filesystem regression")
    def test_windows_long_unicode_atomic_save_load_replace_and_lock(self) -> None:
        directory = self.case_root / ("한글 경로 " + "a" * 70) / ("두 번째 폴더 " + "b" * 70) / ("세 번째 " + "c" * 70)
        self.assertTrue(directory.resolve().is_relative_to(self.root))
        directory.mkdir(parents=True)
        path = directory / "model_checkpoint.pt"
        self.assertGreater(len(str(path)), 260)
        for value in (1, 2):
            experiment.atomic_torch_save({"model_state_dict": {"x": torch.tensor([value])}}, path)
            loaded = experiment.load_torch_checkpoint(path, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            self.assertEqual(loaded["model_state_dict"]["x"].device.type, "cpu")
            self.assertEqual(loaded["model_state_dict"]["x"].item(), value)
            experiment.atomic_write_json(directory / "결과.json", {"value": value})
            experiment.atomic_write_csv(directory / "결과.csv", [{"value": value}])
            experiment.atomic_save_npz(directory / "split.npz", train=np.arange(value))
        self.assertEqual(json.loads((directory / "결과.json").read_text(encoding="utf-8")), {"value": 2})
        with np.load(directory / "split.npz", allow_pickle=False) as result:
            np.testing.assert_array_equal(result["train"], np.arange(2))
        self.assertFalse(list(directory.glob("*.tmp")))
        with experiment.experiment_locks(directory):
            with self.assertRaisesRegex(RuntimeError, "locked"):
                with experiment.experiment_locks(directory):
                    self.fail("A second writer acquired the lock")

    def test_atomic_replace_retries_and_preserves_previous_file_after_failure(self) -> None:
        path = self.case_root / "latest.pt"
        experiment.atomic_torch_save({"x": torch.tensor([1])}, path)
        original_replace = os.replace
        calls = []

        def transient_lock(source, destination):
            calls.append((source, destination))
            self.assertEqual(source.parent, destination.parent)  # 같은 볼륨에서 원자 교체.
            if len(calls) == 1:
                raise PermissionError("transient Windows file lock")
            return original_replace(source, destination)

        with mock.patch.object(os, "replace", side_effect=transient_lock), mock.patch.object(experiment.time, "sleep"):
            experiment.atomic_torch_save({"x": torch.tensor([2])}, path)
        self.assertEqual(len(calls), 2)
        self.assertEqual(experiment.load_torch_checkpoint(path, torch.device("cpu"))["x"].item(), 2)
        before = path.read_bytes()
        with mock.patch.object(os, "replace", side_effect=PermissionError("persistent lock")) as replacing, mock.patch.object(
            experiment.time, "sleep",
        ), self.assertRaises(PermissionError):
            experiment.atomic_torch_save({"x": torch.tensor([3])}, path)
        self.assertEqual(replacing.call_count, 8)
        self.assertEqual(path.read_bytes(), before)
        self.assertFalse(list(self.case_root.glob("*.tmp")))


if __name__ == "__main__":
    unittest.main()
