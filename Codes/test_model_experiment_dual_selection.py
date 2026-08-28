from __future__ import annotations

import contextlib
import copy
import csv
import hashlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest import mock

import numpy as np
import torch

import generate_charge_dataset as generator


_spec = importlib.util.spec_from_file_location(
    "model_experiment_dual_selection_tests", Path(__file__).with_name("ModelExperiment8.5.py")
)
assert _spec is not None and _spec.loader is not None
experiment = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = experiment
_spec.loader.exec_module(experiment)


class DualSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, previous_threads)
        cls.temporary = tempfile.TemporaryDirectory(prefix="dual-selection-tests-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        cls.data_path = cls.root / "physical-test-data.npz"
        generator.save_dataset(
            generator.generate_dataset(sample_count=20, g05_point_count=32, seed=91),
            cls.data_path,
        )
        cls.arrays = experiment.physics.load_dataset(cls.data_path)
        cls.split = experiment.physics.create_data_split(20, 42)
        cls.stats = experiment.physics.calculate_normalization_stats(cls.arrays, cls.split.train)
        cls.datasets = {
            fraction: tuple(
                experiment.physics.prepare_dataset(cls.arrays, indices, cls.stats, fraction)
                for indices in (cls.split.train, cls.split.validation, cls.split.test)
            )
            for fraction in (0.0, 0.75, 1.0)
        }

    def setUp(self) -> None:
        # Leave room for fingerprinted run IDs and atomic temp filenames on Windows.
        self.case_root = self.root / hashlib.sha256(self._testMethodName.encode()).hexdigest()[:8]

    def settings(self, epochs: int = 3) -> experiment.TrainingSettings:
        return experiment.TrainingSettings(
            batch_size=8, max_epochs=epochs, learning_rate=1e-3,
            weight_decay=1e-4, loss_weights=experiment.LossWeights(),
        )

    def config(
        self, model_name: str = "g05_full_reconstruction", fraction: float = 0.75,
        seed: int = 41, epochs: int = 3, protocol: str = "test-protocol",
    ) -> dict:
        return experiment.run_configuration(
            protocol_fingerprint=protocol, code_sha256="test-code",
            spec=experiment.MODEL_REGISTRY[model_name], fraction=fraction,
            g05_count=experiment.physics.g05_count_for_fraction(fraction, 32),
            candidate_count=32, seed=seed, settings=self.settings(epochs),
        )

    def run_model(self, root: Path, config: dict) -> tuple[dict, bool]:
        train, validation, test = self.datasets[config["observation"]["g05_fraction"]]
        with contextlib.redirect_stdout(io.StringIO()):
            return experiment.train_and_evaluate_run(
                spec=experiment.MODEL_REGISTRY[config["model"]["name"]],
                train_dataset=train, validation_dataset=validation, test_dataset=test,
                stats=self.stats, run_config=config, experiment_results_dir=root / "results",
                experiment_checkpoint_dir=root / "checkpoints",
                settings=self.settings(config["training"]["max_epochs"]),
                device=torch.device("cpu"),
            )

    def paths(self, root: Path, config: dict) -> dict[str, Path]:
        return experiment.run_checkpoint_paths(root / "checkpoints" / experiment.run_id_for(config))

    def load(self, root: Path, config: dict, selection: str = "latest") -> dict:
        return experiment.load_torch_checkpoint(self.paths(root, config)[selection], torch.device("cpu"))

    def assert_nested_equal(self, first, second) -> None:
        if isinstance(first, torch.Tensor):
            torch.testing.assert_close(first, second, rtol=0, atol=0)
        elif isinstance(first, np.ndarray):
            np.testing.assert_array_equal(first, second)
        elif isinstance(first, dict):
            self.assertEqual(first.keys(), second.keys())
            for key in first:
                self.assert_nested_equal(first[key], second[key])
        elif isinstance(first, (list, tuple)):
            self.assertEqual(len(first), len(second))
            for a, b in zip(first, second):
                self.assert_nested_equal(a, b)
        else:
            self.assertEqual(first, second)

    @staticmethod
    def scripted_loss(structure: float, global_sign: float) -> experiment.EpochLoss:
        return experiment.EpochLoss(
            total=structure + global_sign, structure=structure, position=structure,
            magnitude=0.0, relative_sign=0.0, global_sign=global_sign,
        )

    def test_independent_selection_saves_complete_states_and_tests_only_after_training(self) -> None:
        root = self.case_root
        config = self.config()
        scores = [(2.0, 0.1), (1.0, 2.0), (1.0, 2.0)]
        snapshots = []
        validation_calls = []
        evaluation_calls = []
        original_epoch = experiment.run_epoch
        original_evaluate = experiment.evaluate_model

        def run_epoch(model, loader, **kwargs):
            self.assertIsNot(loader.dataset, self.datasets[0.75][2])
            actual = original_epoch(model, loader, **kwargs)
            if kwargs.get("optimizer") is not None:
                snapshots.append(experiment.copy_model_state(model))
                return actual
            validation_calls.append(True)
            return self.scripted_loss(*scores[len(validation_calls) - 1])

        def evaluate(model, dataset, stats, **kwargs):
            self.assertEqual(len(validation_calls), 3)
            self.assertIs(dataset, self.datasets[0.75][2])
            selected_index = len(evaluation_calls)  # total=epoch 1; structure=epoch 2
            self.assert_nested_equal(model.state_dict(), snapshots[selected_index])
            actual = original_evaluate(model, dataset, stats, **kwargs)
            evaluation_calls.append(actual)
            return actual

        with mock.patch.object(experiment, "run_epoch", side_effect=run_epoch), mock.patch.object(
            experiment, "evaluate_model", side_effect=evaluate,
        ):
            result, skipped = self.run_model(root, config)
        self.assertFalse(skipped)
        self.assertEqual(len(evaluation_calls), 2)
        self.assertEqual(len(set(self.paths(root, config).values())), 3)
        latest = self.load(root, config)
        experiment.validate_resume_checkpoint(latest, config)
        self.assertEqual(latest["epoch"], 3)
        self.assertEqual((latest["best_total_epoch"], latest["best_structure_epoch"]), (1, 2))
        self.assertEqual((latest["best_total_loss"], latest["best_structure_loss"]), (2.1, 1.0))
        self.assert_nested_equal(latest["model_state_dict"], snapshots[2])
        for index, selection in enumerate(experiment.CHECKPOINT_SELECTIONS):
            checkpoint = self.load(root, config, selection)
            self.assert_nested_equal(checkpoint["model_state_dict"], snapshots[index])
            self.assertEqual(set(checkpoint["model_state_dict"]), set(snapshots[2]))
            evaluation = result["evaluations"][selection]
            for key, value in experiment.run_metadata(config).items():
                self.assertEqual(checkpoint[key], value)
                self.assertEqual(evaluation[key], value)
            self.assertEqual(evaluation["checkpoint_selection"], selection)
            self.assertEqual(evaluation["selection_objective"], f"validation_loss.{selection}")
            self.assertEqual(evaluation["selected_epoch"], index + 1)
            self.assert_nested_equal(evaluation["test_metrics"], evaluation_calls[index])
            self.assertTrue(set(experiment.METRIC_NAMES).issubset(evaluation["test_metrics"]))
        self.assertFalse(result["evaluations"]["structure"]["global_sign_in_selection_objective"])
        self.assertTrue(result["evaluations"]["total"]["global_sign_in_selection_objective"])
        self.assertIn("not optimized", result["evaluations"]["structure"]["global_sign_metrics_note"])
        records = experiment.completed_result_evaluations(result)
        self.assertEqual(len(records), 2)
        for record in records:
            row = experiment.result_to_row(record)
            self.assertEqual(row["best_epoch"], row["selected_epoch"])
            self.assertEqual(row["best_validation_loss"], row["selected_validation_loss"])
            self.assertEqual(Path(row["best_checkpoint"]).name, f"best_{row['checkpoint_selection']}.pt")

    def test_global_sign_validation_changes_do_not_change_structure_selection(self) -> None:
        config = self.config()
        results = []
        states = []
        original_epoch = experiment.run_epoch
        for index, global_losses in enumerate(((0.1, 2.0, 2.0), (3.0, 3.0, 0.0))):
            root = self.case_root / str(index)
            validation_scores = iter(zip((2.0, 1.0, 1.0), global_losses))

            def controlled_epoch(model, loader, **kwargs):
                actual = original_epoch(model, loader, **kwargs)
                return actual if kwargs.get("optimizer") is not None else self.scripted_loss(*next(validation_scores))

            with mock.patch.object(experiment, "run_epoch", side_effect=controlled_epoch):
                result, _ = self.run_model(root, config)
            results.append(result)
            states.append(self.load(root, config))
        self.assertEqual([r["training_result"]["best_total_epoch"] for r in results], [1, 3])
        self.assertEqual([r["training_result"]["best_structure_epoch"] for r in results], [2, 2])
        self.assert_nested_equal(states[0]["model_state_dict"], states[1]["model_state_dict"])
        self.assert_nested_equal(
            states[0]["best_checkpoints"]["structure"]["model_state_dict"],
            states[1]["best_checkpoints"]["structure"]["model_state_dict"],
        )
        self.assert_nested_equal(results[0]["evaluations"]["structure"]["test_metrics"],
                                 results[1]["evaluations"]["structure"]["test_metrics"])

    def test_resume_matches_uninterrupted_training_across_atomic_save_boundaries(self) -> None:
        for model_name in experiment.DEFAULT_MODELS:
            config = self.config(model_name=model_name)
            baseline_root = self.case_root / model_name / "baseline"
            baseline_result, _ = self.run_model(baseline_root, config)
            baseline = self.load(baseline_root, config)
            for phase in ("before_latest", "after_latest", "after_total", "after_final_latest"):
                with self.subTest(model=model_name, phase=phase):
                    root = self.case_root / model_name / phase
                    original_save = experiment.atomic_torch_save

                    def interrupted_save(value, path):
                        if phase == "before_latest" and path.name == "latest.pt" and value["epoch"] == 2:
                            raise InterruptedError("injected before commit")
                        original_save(value, path)
                        if (
                            (phase == "after_latest" and path.name == "latest.pt" and value["epoch"] == 1)
                            or (phase == "after_total" and path.name == "best_total.pt" and value["epoch"] == 1)
                            or (phase == "after_final_latest" and path.name == "latest.pt" and value["epoch"] == 3)
                        ):
                            raise InterruptedError("injected after commit")

                    with mock.patch.object(experiment, "atomic_torch_save", side_effect=interrupted_save):
                        with self.assertRaisesRegex(InterruptedError, "injected"):
                            self.run_model(root, config)
                    partial = self.load(root, config)
                    experiment.validate_resume_checkpoint(partial, config)
                    if phase == "after_latest":
                        self.assertFalse(self.paths(root, config)["total"].exists())
                        self.assertFalse(self.paths(root, config)["structure"].exists())
                    if phase == "after_total":
                        self.assertTrue(self.paths(root, config)["total"].exists())
                        self.assertFalse(self.paths(root, config)["structure"].exists())
                    original_epoch = experiment.run_epoch
                    with mock.patch.object(experiment, "run_epoch", wraps=original_epoch) as epoch_calls:
                        resumed_result, skipped = self.run_model(root, config)
                    self.assertFalse(skipped)
                    self.assertEqual(epoch_calls.call_count, 2 * (3 - partial["epoch"]))
                    resumed = self.load(root, config)
                    for key in (
                        "epoch", "model_state_dict", "optimizer_state_dict", "rng_state",
                        "shuffle_generator_state", "history", "best_checkpoints",
                        "best_total_epoch", "best_total_loss", "best_structure_epoch", "best_structure_loss",
                    ):
                        self.assert_nested_equal(baseline[key], resumed[key])
                    self.assertTrue(resumed_result["training_result"]["resumed"])
                    for selection in experiment.CHECKPOINT_SELECTIONS:
                        self.assert_nested_equal(self.load(baseline_root, config, selection), self.load(root, config, selection))
                        self.assert_nested_equal(
                            baseline_result["evaluations"][selection]["test_metrics"],
                            resumed_result["evaluations"][selection]["test_metrics"],
                        )

    def test_interrupted_second_test_evaluation_resumes_without_training(self) -> None:
        root = self.case_root
        config = self.config(epochs=2)
        original_evaluate = experiment.evaluate_model
        calls = 0

        def interrupted_evaluate(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise InterruptedError("second evaluation stopped")
            return original_evaluate(*args, **kwargs)

        with mock.patch.object(experiment, "evaluate_model", side_effect=interrupted_evaluate):
            with self.assertRaises(InterruptedError):
                self.run_model(root, config)
        result_path = root / "results" / "runs" / experiment.run_id_for(config) / "result.json"
        self.assertFalse(result_path.exists())
        with mock.patch.object(experiment, "run_epoch", side_effect=AssertionError("training already finished")):
            result, skipped = self.run_model(root, config)
        self.assertFalse(skipped)
        self.assertEqual(set(result["evaluations"]), {"total", "structure"})
        self.assertTrue(result_path.exists())
        # A completed run is skipped only when both evaluations are present.
        saved_bytes = result_path.read_bytes()
        with mock.patch.object(experiment, "run_epoch", side_effect=AssertionError("unexpected training")), mock.patch.object(
            experiment, "evaluate_model", side_effect=AssertionError("unexpected evaluation"),
        ):
            repeated, skipped = self.run_model(root, config)
        self.assertTrue(skipped)
        self.assertEqual(result_path.read_bytes(), saved_bytes)
        self.assertEqual(set(repeated["evaluations"]), {"total", "structure"})

    def test_completed_skip_restores_missing_best_files_without_training(self) -> None:
        root = self.case_root
        config = self.config(epochs=1)
        result, _ = self.run_model(root, config)
        paths = self.paths(root, config)
        result_path = root / "results" / "runs" / result["run_id"] / "result.json"
        result_bytes = result_path.read_bytes()
        latest = self.load(root, config)
        for missing in (("total",), ("structure",), ("total", "structure"), ()):
            with self.subTest(missing=missing):
                for selection in experiment.CHECKPOINT_SELECTIONS:
                    if not paths[selection].exists():
                        experiment.atomic_torch_save(latest["best_checkpoints"][selection], paths[selection])
                preserved = {key: path.read_bytes() for key, path in paths.items() if key not in missing}
                for selection in missing:
                    paths[selection].unlink()
                with mock.patch.object(experiment, "run_epoch", side_effect=AssertionError("unexpected training")), mock.patch.object(
                    experiment, "evaluate_model", side_effect=AssertionError("unexpected evaluation"),
                ), mock.patch.object(experiment, "atomic_torch_save", wraps=experiment.atomic_torch_save) as save:
                    _, skipped = self.run_model(root, config)
                self.assertTrue(skipped)
                self.assertEqual(save.call_count, len(missing))
                for selection in missing:
                    self.assert_nested_equal(self.load(root, config, selection), latest["best_checkpoints"][selection])
                for key, contents in preserved.items():
                    self.assertEqual(paths[key].read_bytes(), contents)
                self.assertEqual(result_path.read_bytes(), result_bytes)

    def test_completed_skip_requires_valid_latest_to_restore_missing_best(self) -> None:
        root = self.case_root
        config = self.config(epochs=1)
        result, _ = self.run_model(root, config)
        paths = self.paths(root, config)
        latest = self.load(root, config)
        paths["structure"].unlink()
        for unavailable in (False, True):
            with self.subTest(unavailable=unavailable):
                if unavailable:
                    paths["latest"].unlink()
                else:
                    latest["run_fingerprint"] = "wrong-run"
                    experiment.atomic_torch_save(latest, paths["latest"])
                with mock.patch.object(experiment, "run_epoch", side_effect=AssertionError("unexpected training")), mock.patch.object(
                    experiment, "evaluate_model", side_effect=AssertionError("unexpected evaluation"),
                ):
                    message = "latest checkpoint is missing" if unavailable else "metadata mismatch"
                    with self.assertRaisesRegex(RuntimeError, message):
                        self.run_model(root, config)
                self.assertFalse(paths["structure"].exists())
                saved = json.loads((root / "results" / "runs" / result["run_id"] / "result.json").read_text(encoding="utf-8"))
                self.assertEqual(saved["status"], "completed")

    def test_completed_skip_repairs_status_after_result_commit_interruption(self) -> None:
        root = self.case_root
        config = self.config(epochs=1)
        original_save_status = experiment.save_status

        def interrupt_completion(*args, **kwargs):
            if kwargs["status"] == "completed":
                raise KeyboardInterrupt("interrupted after result commit")
            return original_save_status(*args, **kwargs)

        with mock.patch.object(experiment, "save_status", side_effect=interrupt_completion):
            with self.assertRaises(KeyboardInterrupt) as interrupted:
                self.run_model(root, config)
        experiment.mark_run_failure(
            experiment_results_dir=root / "results", run_config=config,
            status="interrupted", error=interrupted.exception,
        )
        result_dir = root / "results" / "runs" / experiment.run_id_for(config)
        status_path = result_dir / "status.json"
        self.assertEqual(json.loads(status_path.read_text(encoding="utf-8"))["status"], "interrupted")
        preserved = {path: path.read_bytes() for path in (*self.paths(root, config).values(), result_dir / "result.json")}
        with mock.patch.object(experiment, "run_epoch", side_effect=AssertionError("unexpected training")), mock.patch.object(
            experiment, "evaluate_model", side_effect=AssertionError("unexpected evaluation"),
        ):
            result, skipped = self.run_model(root, config)
        self.assertTrue(skipped)
        status = json.loads(status_path.read_text(encoding="utf-8"))
        self.assertEqual(status["status"], "completed")
        self.assertNotIn("error", status)
        for key, value in experiment.best_tracking_fields(result["evaluations"]).items():
            self.assertEqual(status[key], value)
        for path, contents in preserved.items():
            self.assertEqual(path.read_bytes(), contents)

    def test_zero_g05_models_and_both_selection_policies_match_after_training(self) -> None:
        results = []
        trained_models = []
        for model_name in experiment.DEFAULT_MODELS:
            config = self.config(model_name=model_name, fraction=0.0, epochs=2)
            root = self.case_root / model_name
            result, _ = self.run_model(root, config)
            results.append(result)
            for selection in experiment.CHECKPOINT_SELECTIONS:
                model = experiment.MODEL_REGISTRY[model_name].factory()
                model.load_state_dict(self.load(root, config, selection)["model_state_dict"])
                trained_models.append(model)
            self.assert_nested_equal(self.load(root, config, "total")["model_state_dict"],
                                     self.load(root, config, "structure")["model_state_dict"])
            self.assertEqual(result["training_result"]["best_total_epoch"], result["training_result"]["best_structure_epoch"])
            for selection in experiment.CHECKPOINT_SELECTIONS:
                self.assertFalse(config["training"]["checkpoint_selection"][selection]["global_sign_in_selection_objective"])
                self.assertFalse(self.load(root, config, selection)["global_sign_in_selection_objective"])
                self.assertFalse(result["evaluations"][selection]["global_sign_in_selection_objective"])
                for metric in ("global_sign_bce", "global_sign_accuracy", "absolute_sign_accuracy", "signed_pair_accuracy"):
                    self.assertIsNone(result["evaluations"][selection]["test_metrics"][metric])
        with torch.no_grad():
            outputs = [model(*self.datasets[0.0][2].tensors[:3]) for model in trained_models]
        for output in outputs[1:]:
            for field in ("position", "magnitude", "relative_sign_logit", "global_sign_logit"):
                self.assert_nested_equal(getattr(outputs[0], field), getattr(output, field))
        self.assert_nested_equal(results[0]["evaluations"]["structure"]["test_metrics"],
                                 results[1]["evaluations"]["structure"]["test_metrics"])

    def test_zero_global_sign_weight_is_not_in_selection_metadata(self) -> None:
        settings = replace(self.settings(epochs=1), loss_weights=replace(experiment.LossWeights(), global_sign=0.0))
        with mock.patch.object(self, "settings", return_value=settings):
            config = self.config(epochs=1)
            result, _ = self.run_model(self.case_root, config)
        for selection in experiment.CHECKPOINT_SELECTIONS:
            self.assertFalse(config["training"]["checkpoint_selection"][selection]["global_sign_in_selection_objective"])
            self.assertFalse(self.load(self.case_root, config, selection)["global_sign_in_selection_objective"])
            evaluation = result["evaluations"][selection]
            self.assertFalse(evaluation["global_sign_in_selection_objective"])
            self.assertGreater(evaluation["validation_losses"]["global_sign"], 0.0)
            self.assertEqual(evaluation["validation_losses"]["total"], evaluation["validation_losses"]["structure"])
        records = experiment.completed_result_evaluations(result)
        for row in [*(experiment.result_to_row(record) for record in records), *experiment.build_summary_rows(records)]:
            self.assertFalse(row["global_sign_in_selection_objective"])
        for protocol_settings, expected in ((settings, False), (self.settings(epochs=1), None)):
            protocol = experiment.build_protocol(
                data_path=self.data_path, data_sha256=experiment.file_sha256(self.data_path),
                arrays=self.arrays, split=self.split, split_seed=42, stats=self.stats,
                settings=protocol_settings, device=torch.device("cpu"), code_sha256="test-code",
            )
            self.assertIs(protocol["training"]["model_selection"]["total"]["global_sign_in_selection_objective"], expected)
            self.assertFalse(protocol["training"]["model_selection"]["structure"]["global_sign_in_selection_objective"])

    def test_sign_only_structure_trajectory_is_independent_of_fraction(self) -> None:
        checkpoints = []
        results = []
        for fraction in (0.0, 0.75):
            config = self.config(model_name="g05_sign_only", fraction=fraction, epochs=2)
            root = self.case_root / str(fraction)
            result, _ = self.run_model(root, config)
            results.append(result)
            checkpoints.append(self.load(root, config))
        for split_name in ("train", "validation"):
            self.assertEqual(
                [row[split_name]["structure"] for row in checkpoints[0]["history"]],
                [row[split_name]["structure"] for row in checkpoints[1]["history"]],
            )
        prefixes = ("g00_cnn.", "g00_encoder.", "position_head.", "magnitude_head.", "relative_sign_head.")
        for name, value in checkpoints[0]["model_state_dict"].items():
            if name.startswith(prefixes):
                self.assert_nested_equal(value, checkpoints[1]["model_state_dict"][name])
        self.assertEqual(checkpoints[0]["best_structure_epoch"], checkpoints[1]["best_structure_epoch"])
        for metric in experiment.STRUCTURE_METRIC_NAMES:
            self.assertEqual(results[0]["evaluations"]["structure"]["test_metrics"][metric],
                             results[1]["evaluations"]["structure"]["test_metrics"][metric])

    def test_smoke_only_cli_checks_routes_and_resume_without_full_training(self) -> None:
        root = self.case_root
        argv = [
            "ModelExperiment8.5.py", "--data", str(self.data_path), "--smoke-only",
            "--fractions", "0,0.75", "--seeds", "41,42,43", "--epochs", "300",
            "--device", "cpu", "--results-root", str(root / "results"),
            "--checkpoint-root", str(root / "checkpoints"),
        ]
        with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()), mock.patch.object(
            experiment, "run_epoch", side_effect=AssertionError("full training is forbidden"),
        ), mock.patch.object(experiment, "evaluate_model", side_effect=AssertionError("no test data in smoke tests")):
            args = experiment.parse_args()
            self.assertEqual(args.seeds, (41, 42, 43))
            self.assertEqual(args.epochs, 300)
            self.assertNotEqual(args.experiment_name, "g05_routing_comparison_v1")
            experiment.main()
        self.assertFalse(root.exists())

    def test_split_train_only_normalization_and_nested_sensor_prefix_are_unchanged(self) -> None:
        expected = np.random.default_rng(42).permutation(20)
        self.assert_nested_equal(self.split.train, expected[:16])
        self.assert_nested_equal(self.split.validation, expected[16:18])
        self.assert_nested_equal(self.split.test, expected[18:])
        modified = replace(self.arrays, g00=self.arrays.g00.copy(), g05=self.arrays.g05.copy(), target=self.arrays.target.copy())
        held_out = np.concatenate((self.split.validation, self.split.test))
        modified.g00[held_out] += 100
        modified.g05[held_out, :, 2] += 100
        modified.target[held_out] += 100
        self.assert_nested_equal(asdict(self.stats), asdict(experiment.physics.calculate_normalization_stats(modified, self.split.train)))
        masks = [experiment.physics.create_g05_mask(2, 32, fraction) for fraction in experiment.physics.G05_FRACTIONS]
        self.assertEqual([int(mask[0].sum()) for mask in masks], [0, 3, 8, 16, 24, 32])
        for lower, higher in zip(masks, masks[1:]):
            self.assertTrue(np.all(lower <= higher))
        for mask in masks:
            count = int(mask[0].sum())
            self.assertTrue(np.all(mask[:, :count] == 1))
            self.assertTrue(np.all(mask[:, count:] == 0))
        experiment.physics.verify_physical_consistency(self.arrays)

    def test_same_seed_gives_same_initial_state_and_batch_order(self) -> None:
        models = []
        orders = []
        dataset = torch.utils.data.TensorDataset(torch.arange(17))
        for model_name in experiment.DEFAULT_MODELS:
            experiment.set_reproducibility(41)
            models.append(experiment.MODEL_REGISTRY[model_name].factory())
            loader = experiment.make_loader(dataset, batch_size=4, device=torch.device("cpu"), shuffle=True,
                                            generator=torch.Generator().manual_seed(41))
            orders.append([torch.cat([batch[0] for batch in loader]) for _ in range(3)])
        self.assertTrue(experiment.state_dicts_are_identical(*models))
        self.assertEqual(experiment.parameter_counts(models[0]), experiment.parameter_counts(models[1]))
        self.assert_nested_equal(orders[0], orders[1])

    def fake_result(self, model_name: str, seed: int, protocol: str = "test-protocol") -> dict:
        """Synthetic metrics for report arithmetic only; never research results."""
        config = self.config(model_name=model_name, seed=seed, protocol=protocol)
        evaluations = {}
        for selection, epoch in (("total", 1), ("structure", 2)):
            is_b = model_name == "g05_full_reconstruction"
            error = (12.0 if is_b else 10.0) if selection == "total" else (3.0 if is_b else 5.0)
            error += (seed - 41) * 0.1
            accuracy = (0.9 if is_b else 0.8) if selection == "total" else (0.6 if is_b else 0.7)
            metrics = {name: error if name in experiment.LOWER_IS_BETTER else accuracy for name in experiment.METRIC_NAMES}
            metrics.update(position_mae=[error] * 6, observed_sample_fraction=1.0, observations_per_sample=24.0)
            loss = self.scripted_loss(2.0, 1.0)
            evaluations[selection] = {
                **experiment.run_metadata(config), **config["training"]["checkpoint_selection"][selection],
                "selected_epoch": epoch, "selected_validation_loss": getattr(loss, selection),
                "validation_losses": asdict(loss), "test_metrics": metrics,
                "checkpoint_path": str(self.root / "fake-checkpoints" / experiment.run_id_for(config) / f"best_{selection}.pt"),
            }
        return {
            "result_schema_version": experiment.RESULT_SCHEMA_VERSION,
            **experiment.run_metadata(config), "status": "completed", "configuration": config,
            "parameter_count": {"total": 1, "trainable": 1},
            "training_result": {**experiment.best_tracking_fields(evaluations), "elapsed_seconds": 1.0},
            "evaluations": evaluations,
        }

    def test_reports_separate_policies_preserve_three_seed_counts_and_improvement_sign(self) -> None:
        root = self.case_root
        for model_name in experiment.DEFAULT_MODELS:
            for seed in (41, 42, 43):
                result = self.fake_result(model_name, seed)
                experiment.atomic_write_json(root / "runs" / result["run_id"] / "result.json", result)
        # A legacy result has a different protocol and must not be counted as new.
        experiment.atomic_write_json(root / "runs" / "legacy" / "result.json", {
            "status": "completed", "configuration": {"protocol_fingerprint": "legacy-protocol"},
            "test_metrics": {"mean_position_mae": -1000},
        })
        experiment.refresh_reports(root, "test-protocol")

        def read_csv(name):
            with (root / name).open(encoding="utf-8", newline="") as handle:
                return list(csv.DictReader(handle))

        runs, summaries, pairs = (read_csv(name) for name in ("runs.csv", "summary.csv", "pairwise_comparisons.csv"))
        self.assertEqual(len(runs), 12)
        self.assertEqual(len(summaries), 4)
        self.assertEqual(len(pairs), 2 * len(experiment.METRIC_NAMES))
        for row in runs:
            self.assertIn(row["checkpoint_selection"], experiment.CHECKPOINT_SELECTIONS)
            for key in ("selection_objective", "selected_epoch", "selected_validation_loss", "model",
                        "g05_fraction", "g05_count_per_sample", "seed", "run_fingerprint", "protocol_fingerprint"):
                self.assertTrue(row[key], key)
        summary_by_key = {(row["checkpoint_selection"], row["model"]): row for row in summaries}
        self.assertAlmostEqual(float(summary_by_key[("total", "g05_sign_only")]["mean_position_mae_mean"]), 10.1)
        self.assertAlmostEqual(float(summary_by_key[("structure", "g05_sign_only")]["mean_position_mae_mean"]), 5.1)
        for row in summaries:
            self.assertEqual(row["run_count"], "3")
            self.assertEqual(row["seeds"], "41,42,43")
            self.assertEqual(set(json.loads(row["selected_epochs_by_seed"])), {"41", "42", "43"})
        pair_by_key = {(row["checkpoint_selection"], row["metric"]): row for row in pairs}
        for selection, improvement in (("total", -2.0), ("structure", 2.0)):
            for metric in ("mean_position_mae", "mean_position_3d_error", *experiment.POSITION_MAE_NAMES):
                row = pair_by_key[(selection, metric)]
                self.assertEqual((row["model_a"], row["model_b"]), experiment.DEFAULT_MODELS)
                self.assertEqual(row["paired_seed_count"], "3")
                self.assertAlmostEqual(float(row["improvement_b_over_a_mean"]), improvement)
                for value in json.loads(row["improvement_b_over_a_by_seed"]).values():
                    self.assertAlmostEqual(value, improvement)
        self.assertAlmostEqual(float(pair_by_key[("total", "relative_sign_accuracy")]["improvement_b_over_a_mean"]), 0.1)
        self.assertAlmostEqual(float(pair_by_key[("structure", "relative_sign_accuracy")]["improvement_b_over_a_mean"]), -0.1)
        self.assertEqual(pair_by_key[("structure", "global_sign_accuracy")]["metric_role"], "secondary")
        self.assertEqual(pair_by_key[("structure", "mean_position_mae")]["primary_research_metric"], "True")

    def test_reports_remain_unchanged_if_any_result_is_unreadable(self) -> None:
        root = self.case_root
        for model_name in experiment.DEFAULT_MODELS:
            result = self.fake_result(model_name, 41)
            experiment.atomic_write_json(root / "runs" / result["run_id"] / "result.json", result)
        experiment.refresh_reports(root, "test-protocol")
        reports = {root / name: (root / name).read_bytes() for name in ("runs.csv", "summary.csv", "pairwise_comparisons.csv")}
        unreadable_path = root / "runs" / result["run_id"] / "result.json"
        original_open = Path.open
        for failure in (PermissionError("temporary Windows lock"), json.JSONDecodeError("truncated JSON", "{", 1)):
            with self.subTest(failure=type(failure).__name__):
                def guarded_open(path, *args, **kwargs):
                    if path == unreadable_path:
                        raise failure
                    return original_open(path, *args, **kwargs)

                with mock.patch.object(Path, "open", guarded_open), contextlib.redirect_stdout(io.StringIO()):
                    experiment.refresh_reports(root, "test-protocol")
                for path, contents in reports.items():
                    self.assertEqual(path.read_bytes(), contents)

    def test_reports_clear_stale_rows_when_results_are_removed(self) -> None:
        root = self.case_root
        result_paths = []
        for model_name in experiment.DEFAULT_MODELS:
            result = self.fake_result(model_name, 41)
            path = root / "runs" / result["run_id"] / "result.json"
            experiment.atomic_write_json(path, result)
            result_paths.append(path)
        experiment.refresh_reports(root, "test-protocol")
        headers = {name: (root / name).read_text(encoding="utf-8").splitlines()[0] for name in ("runs.csv", "summary.csv", "pairwise_comparisons.csv")}
        for removed, expected_counts in zip(reversed(result_paths), ((2, 2, 0), (0, 0, 0))):
            removed.unlink()
            experiment.refresh_reports(root, "test-protocol")
            for name, count in zip(headers, expected_counts):
                with (root / name).open(encoding="utf-8", newline="") as handle:
                    self.assertEqual(len(list(csv.DictReader(handle))), count, name)
                self.assertEqual((root / name).read_text(encoding="utf-8").splitlines()[0], headers[name])

    def test_reports_never_pair_different_selections_protocols_or_seeds(self) -> None:
        a = experiment.completed_result_evaluations(self.fake_result("g05_sign_only", 41))
        b = experiment.completed_result_evaluations(self.fake_result("g05_full_reconstruction", 41))
        b_other_seed = experiment.completed_result_evaluations(self.fake_result("g05_full_reconstruction", 42))
        b_other_protocol = experiment.completed_result_evaluations(self.fake_result("g05_full_reconstruction", 41, "other"))
        self.assertEqual(experiment.build_pairwise_rows([a[0], b[1]]), [])
        self.assertEqual(experiment.build_pairwise_rows([*a, *b_other_seed]), [])
        self.assertEqual(experiment.build_pairwise_rows([*a, *b_other_protocol]), [])
        self.assertEqual(len(experiment.build_summary_rows([*a, *b_other_protocol])), 4)
        for builder in (experiment.build_pairwise_rows, experiment.build_summary_rows):
            with self.assertRaisesRegex(RuntimeError, "Duplicate"):
                builder([a[0], a[0]])

    def test_legacy_or_incomplete_results_are_not_relabelled_as_dual_selection(self) -> None:
        result = self.fake_result("g05_sign_only", 41)
        legacy = {key: value for key, value in result.items() if key not in ("result_schema_version", "evaluations")}
        legacy["test_metrics"] = {"mean_position_mae": 1.0}
        with self.assertRaisesRegex(RuntimeError, "legacy total-only"):
            experiment.completed_result_evaluations(legacy)
        del result["evaluations"]["structure"]
        with self.assertRaisesRegex(RuntimeError, "both evaluations"):
            experiment.completed_result_evaluations(result)
        with self.assertRaisesRegex(RuntimeError, "Legacy best.pt/latest.pt"):
            experiment.validate_resume_checkpoint({"epoch": 3, "best_epoch": 1}, self.config())

    def test_resume_rejects_inconsistent_best_tracker_and_partial_model_state(self) -> None:
        root = self.case_root
        config = self.config(epochs=1)
        self.run_model(root, config)
        latest = self.load(root, config)
        inconsistent = copy.deepcopy(latest)
        inconsistent["best_structure_loss"] += 1
        with self.assertRaisesRegex(RuntimeError, "best tracker mismatch"):
            experiment.validate_resume_checkpoint(inconsistent, config)
        # Replace the mapping instead of mutating it: same-epoch states may share storage.
        incomplete = copy.deepcopy(latest)
        incomplete["best_checkpoints"]["structure"]["model_state_dict"] = {
            name: tensor for name, tensor in latest["model_state_dict"].items()
            if not name.startswith("global_sign_head.")
        }
        with self.assertRaisesRegex(RuntimeError, "not a complete model state"):
            experiment.validate_resume_checkpoint(incomplete, config)

    def test_experiment_name_guard_preserves_legacy_protocol_file(self) -> None:
        root = self.case_root
        protocol_path = root / "results" / "protocol.json"
        experiment.atomic_write_json(protocol_path, {"protocol_fingerprint": "legacy-fingerprint"})
        before = protocol_path.read_bytes()
        protocol = experiment.build_protocol(
            data_path=self.data_path, data_sha256=experiment.file_sha256(self.data_path),
            arrays=self.arrays, split=self.split, split_seed=42, stats=self.stats,
            settings=self.settings(), device=torch.device("cpu"), code_sha256="test-code",
        )
        with self.assertRaisesRegex(RuntimeError, "new --experiment-name"):
            experiment.initialize_experiment_artifacts(
                experiment_results_dir=root / "results", experiment_checkpoint_dir=root / "checkpoints",
                protocol=protocol, split=self.split, stats=self.stats,
                selected_specs=[experiment.MODEL_REGISTRY[name] for name in experiment.DEFAULT_MODELS],
            )
        self.assertEqual(protocol_path.read_bytes(), before)
        self.assertFalse((root / "checkpoints").exists())


if __name__ == "__main__":
    unittest.main()
