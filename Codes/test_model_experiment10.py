"""v9 회귀 시나리오를 v10에도 적용하고 정규화/종료/재개 경계를 추가 검증한다.

기존 기능 보존 테스트에서는 dropout=0, patience=0을 명시한다. 아래 별도 테스트는
드롭아웃이 켜진 실제 optimizer 업데이트와 CPU/CUDA 중단 재개까지 확인한다.
테스트용 20개 합성 샘플의 지표는 실제 데이터의 성능 주장에 사용하지 않는다.
"""

from __future__ import annotations

import contextlib
import copy
import csv
import hashlib
import io
import itertools
import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from torch.utils.data import TensorDataset

import ModelExperiment10 as experiment
import ModelExperiment9 as previous_experiment
import NewLearning9 as physics
import generate_charge_dataset as generator


class FiveChargeExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, previous_threads)
        cls.temporary = tempfile.TemporaryDirectory(prefix="m10-tests-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        cls.data_path = cls.root / "five.npz"
        generator.save_dataset(generator.generate_dataset(sample_count=20, g05_point_count=32,
                                                          seed=91, charge_count=5), cls.data_path)
        cls.arrays = physics.load_dataset(cls.data_path)
        cls.split = physics.create_data_split(20)
        cls.stats = physics.calculate_normalization_stats(cls.arrays, cls.split.train)
        cls.datasets = {fraction: tuple(physics.prepare_dataset(cls.arrays, getattr(cls.split, name), cls.stats, fraction)
                                       for name in ("train", "validation", "test")) for fraction in (0.0, 0.75, 1.0)}

    def setUp(self) -> None:
        self.case_root = self.root / hashlib.sha256(self._testMethodName.encode()).hexdigest()[:8]

    def protocol(self, epochs: int = 3, weights: physics.LossWeights = physics.LossWeights(),
                 device: torch.device = torch.device("cpu")) -> dict:
        return experiment.build_protocol(data_path=self.data_path, arrays=self.arrays, split=self.split,
                                         stats=self.stats, settings=physics.TrainingSettings(max_epochs=epochs, batch_size=8),
                                         regularization=experiment.RegularizationSettings(structure_dropout=0.0,
                                                                                          early_stopping_patience=0),
                                         weights=weights, device=device)

    def config(self, model_name: str = "g05_full_reconstruction", fraction: float = 0.75,
               seed: int = 42, epochs: int = 3, weights: physics.LossWeights = physics.LossWeights(),
               device: torch.device = torch.device("cpu")) -> dict:
        return experiment.run_configuration(self.protocol(epochs, weights, device), model_name=model_name, fraction=fraction, seed=seed)

    def run_model(self, root: Path, config: dict, device: torch.device = torch.device("cpu")) -> tuple[dict, bool]:
        with contextlib.redirect_stdout(io.StringIO()):
            return experiment.train_and_evaluate_run(run_config=config, datasets=self.datasets[config["observation"]["g05_fraction"]],
                                                     experiment_results_dir=root / "results", experiment_checkpoint_dir=root / "checkpoints",
                                                     device=device)

    def paths(self, root: Path, config: dict) -> dict[str, Path]:
        return experiment.run_checkpoint_paths(root / "checkpoints" / experiment.run_id_for(config))

    def load(self, root: Path, config: dict, selection: str = "latest") -> dict:
        return experiment.load_torch_checkpoint(self.paths(root, config)[selection], torch.device("cpu"))

    def evaluation_trial(self, protocol: dict) -> tuple[Path, Path, list[str]]:
        results = self.case_root / "results" / "trial"
        checkpoints = self.case_root / "checkpoints" / "trial"
        experiment.initialize_experiment_artifacts(experiment_results_dir=results, experiment_checkpoint_dir=checkpoints,
                                                  protocol=protocol, split=self.split)
        args = ["--evaluate-only", "--device", "cpu", "--experiment-name", "trial",
                "--results-root", str(results.parent), "--checkpoint-root", str(checkpoints.parent)]
        return results, checkpoints, args

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

    def test_same_capacity_initialization_and_baseline_training_trajectory(self) -> None:
        self.assertIs(experiment.run_epoch, physics.run_epoch)
        self.assertIs(experiment.evaluate_model, physics.evaluate_model)
        models = []
        for factory in (physics.ChargeNet, experiment.MODEL_REGISTRY["g05_sign_only"].factory,
                        experiment.MODEL_REGISTRY["g05_full_reconstruction"].factory):
            experiment.set_reproducibility(42)
            models.append(factory())
        baseline, sign_only, full = models
        self.assertEqual(experiment.parameter_counts(sign_only), experiment.parameter_counts(full))
        self.assert_nested_equal(sign_only.state_dict(), full.state_dict())
        self.assertEqual([n for n, _ in sign_only.named_modules()], [n for n, _ in full.named_modules()])
        for name, value in baseline.state_dict().items():
            self.assert_nested_equal(value, sign_only.state_dict()[name])
        batch = tuple(t[:4] for t in self.datasets[0.75][0].tensors)
        for model in models[1:]:
            experiment.assert_outputs_close(baseline(*batch[:3]), model(*batch[:3]))
        optimizers = [torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4) for m in models[:2]]
        loaders = [physics.create_data_loader(self.datasets[0.75][0], 8, shuffle=True, seed=42, device=torch.device("cpu")) for _ in optimizers]
        for _ in range(3):
            expected = physics.run_epoch(baseline, loaders[0], optimizers[0])
            actual = experiment.run_epoch(sign_only, loaders[1], optimizers[1])
            self.assertEqual(asdict(expected), asdict(actual))
            for name, value in baseline.state_dict().items():
                self.assert_nested_equal(value, sign_only.state_dict()[name])
        self.assertEqual(physics.evaluate_model(baseline, self.datasets[0.75][2], self.stats),
                         experiment.evaluate_model(sign_only, self.datasets[0.75][2], self.stats))

    def test_gradient_routes_masking_and_exact_sign_symmetry_after_update(self) -> None:
        batch = tuple(t[:4].clone() for t in self.datasets[0.75][0].tensors)
        g00, g05, mask, position, charge = batch
        for name, spec in experiment.MODEL_REGISTRY.items():
            with self.subTest(model=name):
                experiment.set_reproducibility(42)
                model = spec.factory()
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
                loss = physics.calculate_losses(model(g00, g05, mask), position, charge, mask)
                loss.structure.backward()
                experiment.assert_no_gradient(model, ("g05_encoder.", "global_sign_head."))
                if name == "g05_sign_only":
                    experiment.assert_no_gradient(model, ("structure_context.",))
                else:
                    self.assertTrue(experiment.has_nonzero_gradient(model, ("structure_context.2.",)))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                points = g05.clone().requires_grad_()
                physics.calculate_losses(model(g00, points, mask), position, charge, mask).structure.backward()
                if name == "g05_full_reconstruction":
                    self.assertTrue(experiment.has_nonzero_gradient(model, ("g05_encoder.",)))
                    self.assertTrue(experiment.has_nonzero_gradient(model, ("structure_context.0.",)))
                    self.assertTrue(torch.any(points.grad[:, :, 2] != 0))
                    self.assertTrue(torch.all(points.grad[~mask.bool().expand_as(points)] == 0))
                else:
                    self.assertIsNone(points.grad)
                    experiment.assert_no_gradient(model, ("g05_encoder.", "structure_context."))
                experiment.assert_no_gradient(model, ("global_sign_head.",))
                optimizer.zero_grad(set_to_none=True)
                physics.calculate_losses(model(g00, g05, mask), position, charge, mask).global_sign.backward()
                for parameter_name, parameter in model.named_parameters():
                    if parameter_name.startswith(experiment.STRUCTURE_PREFIXES):
                        self.assertIsNone(parameter.grad, parameter_name)
                self.assertTrue(experiment.has_nonzero_gradient(model, ("global_sign_head.",)))
                optimizer.step()
                with torch.no_grad():
                    output = model(g00, g05, mask)
                    flipped = model(g00, g05 * torch.tensor([1, 1, -1]), mask)
                    for field in experiment.OUTPUT_FIELDS[:-1]:
                        torch.testing.assert_close(getattr(output, field), getattr(flipped, field), rtol=1e-6, atol=1e-7)
                    torch.testing.assert_close(output.global_sign_logit, -flipped.global_sign_logit, rtol=1e-6, atol=1e-7)
                    self.assert_nested_equal(output.global_sign_logit, model(g00 + 100, g05, mask).global_sign_logit)
                    hidden = g05.masked_fill(~mask.bool().expand_as(g05), float("nan"))
                    experiment.assert_outputs_close(output, model(g00, hidden, mask))
                    order = torch.randperm(g05.shape[1])
                    experiment.assert_outputs_close(output, model(g00, g05[:, order], mask[:, order]), exact=False)

    def test_zero_g05_training_and_both_selected_states_match(self) -> None:
        results, latest = [], []
        for index, name in enumerate(experiment.DEFAULT_MODELS):
            config = self.config(model_name=name, fraction=0.0, epochs=2)
            root = self.case_root / str(index)
            result, _ = self.run_model(root, config)
            results.append(result)
            latest.append(self.load(root, config))
            self.assert_nested_equal(self.load(root, config, "total")["model_state_dict"],
                                     self.load(root, config, "structure")["model_state_dict"])
            for selection in experiment.CHECKPOINT_SELECTIONS:
                evaluation = result["evaluations"][selection]
                self.assertFalse(evaluation["global_sign_in_selection_objective"])
                self.assertEqual(evaluation["validation_losses"]["total"], evaluation["validation_losses"]["structure"])
                for metric in ("global_sign_accuracy", "global_sign_bce", "absolute_sign_accuracy", "absolute_sign_set_accuracy"):
                    self.assertIsNone(evaluation["test_metrics"][metric])
                self.assertEqual(evaluation["test_metrics"]["charge_mae"], evaluation["test_metrics"]["global_invariant_charge_mae"])
        for key in ("model_state_dict", "optimizer_state_dict", "history", "shuffle_generator_state"):
            self.assert_nested_equal(latest[0][key], latest[1][key])
        for selection in experiment.CHECKPOINT_SELECTIONS:
            self.assertEqual(results[0]["evaluations"][selection]["test_metrics"], results[1]["evaluations"][selection]["test_metrics"])

    def test_sign_only_structure_trajectory_is_independent_of_g05_fraction(self) -> None:
        checkpoints, results = [], []
        for index, fraction in enumerate((0.0, 0.75, 1.0)):
            config = self.config(model_name="g05_sign_only", fraction=fraction, epochs=2)
            root = self.case_root / str(index)
            result, _ = self.run_model(root, config)
            results.append(result)
            checkpoints.append(self.load(root, config))
        for checkpoint, result in zip(checkpoints[1:], results[1:]):
            for phase in ("train", "validation"):
                self.assertEqual([row[phase]["structure"] for row in checkpoints[0]["history"]],
                                 [row[phase]["structure"] for row in checkpoint["history"]])
            for name, value in checkpoints[0]["model_state_dict"].items():
                if name.startswith(physics.STRUCTURE_PREFIXES):
                    self.assert_nested_equal(value, checkpoint["model_state_dict"][name])
            self.assertEqual(checkpoints[0]["best_structure_epoch"], checkpoint["best_structure_epoch"])
            for metric in experiment.STRUCTURE_METRIC_NAMES:
                self.assertEqual(results[0]["evaluations"]["structure"]["test_metrics"][metric],
                                 result["evaluations"]["structure"]["test_metrics"][metric])

    def test_target_permutations_matching_and_relative_sign_protocol(self) -> None:
        batch = tuple(t[:2] for t in self.datasets[0.75][0].tensors)
        g00, g05, mask, position, charge = batch
        weights = physics.LossWeights(position=1.7, magnitude=0.8, relative_sign=0.35, global_sign=0.4)
        for spec in experiment.MODEL_REGISTRY.values():
            model = spec.factory()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            physics.calculate_losses(model(g00, g05, mask), position, charge, mask, weights).structure.backward()
            optimizer.step()
            output = model(g00, g05, mask)
            expected = physics.calculate_losses(output, position, charge, mask, weights)
            for permutation in itertools.permutations(range(5)):
                actual = physics.calculate_losses(output, position[:, list(permutation)], charge[:, list(permutation)], mask, weights)
                for field in physics.EpochLoss.__dataclass_fields__:
                    torch.testing.assert_close(getattr(expected, field), getattr(actual, field), rtol=1e-6, atol=1e-6)
            flipped = model(g00, g05 * torch.tensor([1, 1, -1]), mask)
            reversed_loss = physics.calculate_losses(flipped, position, -charge, mask, weights)
            torch.testing.assert_close(expected.total, reversed_loss.total, rtol=1e-6, atol=1e-6)
            relative, global_target = physics.canonical_sign_targets(charge)
            self.assert_nested_equal(relative * 2 - 1, torch.sign(charge) * torch.sign(charge).prod(dim=1)[:, None])
            self.assert_nested_equal(global_target * 2 - 1, torch.sign(charge).prod(dim=1))
            self.assert_nested_equal(physics.decode_relative_signs(output.relative_sign_logit).prod(dim=1), torch.ones(len(g00)))
            pair_cost = physics.matching_cost(output, position, charge, weights)
            assignment = physics.minimum_cost_assignment(pair_cost)
            for cost, chosen in zip(pair_cost.detach(), assignment):
                self.assertEqual(sorted(chosen.tolist()), list(range(5)))
                oracle = min(sum(float(cost[i, j]) for i, j in enumerate(p)) for p in itertools.permutations(range(5)))
                self.assertAlmostEqual(sum(float(cost[i, j]) for i, j in enumerate(chosen)), oracle, places=6)
            changed = replace(output, global_sign_logit=output.global_sign_logit + 100)
            self.assert_nested_equal(pair_cost, physics.matching_cost(changed, position, charge, replace(weights, global_sign=900)))
        self.assertEqual(tuple(physics.relative_sign_patterns(torch.device("cpu"), torch.float32).shape), (16, 5))

    def test_evaluation_is_permutation_invariant_for_both_models(self) -> None:
        for spec in experiment.MODEL_REGISTRY.values():
            model = spec.factory()
            for fraction in (0.0, 0.75):
                dataset = self.datasets[fraction][2]
                g00, g05, mask, position, charge = dataset.tensors
                order = [4, 2, 1, 0, 3]
                shuffled = TensorDataset(g00, g05, mask, position[:, order], charge[:, order])
                expected = experiment.evaluate_model(model, dataset, self.stats, batch_size=2)
                actual = experiment.evaluate_model(model, shuffled, self.stats, batch_size=2)
                self.assertEqual(set(expected), set(experiment.METRIC_NAMES) | {"observed_sample_fraction", "observations_per_sample"})
                for key in expected:
                    if expected[key] is None:
                        self.assertIsNone(actual[key])
                    else:
                        self.assertAlmostEqual(actual[key], expected[key], places=6)

    def test_split_normalization_and_fixed_prefixes_preserve_baseline(self) -> None:
        rng = np.random.default_rng(physics.DATA_SPLIT_SEED)
        expected_indices = rng.permutation(len(self.arrays.target))
        self.assert_nested_equal(self.split.train, expected_indices[:16])
        self.assert_nested_equal(self.split.validation, expected_indices[16:18])
        self.assert_nested_equal(self.split.test, expected_indices[18:])
        self.assertEqual(self.stats.position_mean.shape, (3,))
        changed = replace(self.arrays, target=self.arrays.target.copy(), g00=self.arrays.g00.copy(), g05=self.arrays.g05.copy())
        held_out = np.concatenate((self.split.validation, self.split.test))
        for array in (changed.target, changed.g00, changed.g05):
            array[held_out] += 1000
        self.assertEqual(physics.calculate_normalization_stats(changed, self.split.train).to_dict(), self.stats.to_dict())
        masks = [self.datasets[f][0].tensors[2] for f in (0.0, 0.75, 1.0)]
        for mask, count in zip(masks, (0, 24, 32)):
            self.assertTrue(torch.all(mask[:, :count] == 1))
            self.assertTrue(torch.all(mask[:, count:] == 0))
        self.assertTrue(torch.all(masks[0] <= masks[1]))
        self.assertTrue(torch.all(masks[1] <= masks[2]))
        self.assertEqual(physics.g05_count_for_fraction(1.0, 32), 32)

    def test_separate_minima_save_complete_epoch_states_and_test_only_after_training(self) -> None:
        config = self.config()
        root = self.case_root
        snapshots, validation_calls, evaluations = [], [], []
        original_epoch, original_evaluate = experiment.run_epoch, experiment.evaluate_model
        scores = ((2.0, 0.1), (1.0, 2.0), (1.0, 2.0))

        def run_epoch(model, loader, optimizer=None, weights=physics.LossWeights()):
            self.assertIsNot(loader.dataset, self.datasets[0.75][2])
            actual = original_epoch(model, loader, optimizer, weights)
            if optimizer is not None:
                snapshots.append(experiment.copy_model_state(model))
                return actual
            structure, global_sign = scores[len(validation_calls)]
            validation_calls.append(True)
            return physics.EpochLoss(structure + global_sign, structure, structure, 0.0, 0.0, global_sign)

        def evaluate(model, dataset, stats, **kwargs):
            self.assertEqual(len(validation_calls), 3)
            self.assertIs(dataset, self.datasets[0.75][2])
            self.assert_nested_equal(model.state_dict(), snapshots[len(evaluations)])
            value = original_evaluate(model, dataset, stats, **kwargs)
            evaluations.append(value)
            return value

        with mock.patch.object(experiment, "run_epoch", side_effect=run_epoch), mock.patch.object(experiment, "evaluate_model", side_effect=evaluate):
            result, skipped = self.run_model(root, config)
        self.assertFalse(skipped)
        self.assertEqual(len(evaluations), 2)
        self.assertEqual(len(set(self.paths(root, config).values())), 3)
        latest = self.load(root, config)
        experiment.validate_resume_checkpoint(latest, config)
        self.assertEqual((latest["best_total_epoch"], latest["best_structure_epoch"]), (1, 2))
        self.assert_nested_equal(latest["model_state_dict"], snapshots[2])
        for index, selection in enumerate(experiment.CHECKPOINT_SELECTIONS):
            checkpoint = self.load(root, config, selection)
            self.assert_nested_equal(checkpoint["model_state_dict"], snapshots[index])
            self.assertEqual(result["evaluations"][selection]["selected_epoch"], index + 1)
            loaded, stats, _ = experiment.load_trained_model(self.paths(root, config)[selection])
            self.assert_nested_equal(loaded.state_dict(), snapshots[index])
            self.assertEqual(stats.to_dict(), self.stats.to_dict())
        self.assertFalse(result["evaluations"]["structure"]["global_sign_in_selection_objective"])
        self.assertTrue(result["evaluations"]["total"]["global_sign_in_selection_objective"])

    def test_resume_is_exact_across_atomic_save_boundaries(self) -> None:
        for model_index, model_name in enumerate(experiment.DEFAULT_MODELS):
            config = self.config(model_name=model_name)
            baseline_root = self.case_root / str(model_index) / "base"
            baseline_result, _ = self.run_model(baseline_root, config)
            baseline = self.load(baseline_root, config)
            for index, boundary in enumerate(("latest_before", "latest_after", "history_before", "history_after")):
                with self.subTest(model=model_name, boundary=boundary):
                    root = self.case_root / str(model_index) / str(index)
                    original_torch, original_json = experiment.atomic_torch_save, experiment.atomic_write_json

                    def torch_save(value, path):
                        target = path.name == "latest.pt" and value["epoch"] == 2 and boundary.startswith("latest")
                        if target and boundary.endswith("before"):
                            raise InterruptedError(boundary)
                        original_torch(value, path)
                        if target and boundary.endswith("after"):
                            raise InterruptedError(boundary)

                    def json_save(path, value):
                        target = path.name == "history.json" and len(value) == 2 and boundary.startswith("history")
                        if target and boundary.endswith("before"):
                            raise InterruptedError(boundary)
                        original_json(path, value)
                        if target and boundary.endswith("after"):
                            raise InterruptedError(boundary)

                    with mock.patch.object(experiment, "atomic_torch_save", side_effect=torch_save), mock.patch.object(
                        experiment, "atomic_write_json", side_effect=json_save,
                    ):
                        with self.assertRaises(InterruptedError):
                            self.run_model(root, config)
                    resumed_result, skipped = self.run_model(root, config)
                    self.assertFalse(skipped)
                    self.assertTrue(resumed_result["training_result"]["resumed"])
                    resumed = self.load(root, config)
                    for key in ("model_state_dict", "optimizer_state_dict", "rng_state", "shuffle_generator_state", "history", "best_checkpoints"):
                        self.assert_nested_equal(baseline[key], resumed[key])
                    for selection in experiment.CHECKPOINT_SELECTIONS:
                        self.assert_nested_equal(self.load(baseline_root, config, selection), self.load(root, config, selection))
                        self.assertEqual(baseline_result["evaluations"][selection]["test_metrics"], resumed_result["evaluations"][selection]["test_metrics"])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_full_model_resume_matches_uninterrupted_training(self) -> None:
        device = torch.device("cuda")
        config = self.config(epochs=2, device=device)
        baseline_root, root = self.case_root / "base", self.case_root / "resumed"
        baseline_result, _ = self.run_model(baseline_root, config, device)
        original_save = experiment.atomic_torch_save

        def interrupt(value, path):
            original_save(value, path)
            if path.name == "latest.pt" and value["epoch"] == 1:
                raise InterruptedError("after CUDA epoch commit")

        with mock.patch.object(experiment, "atomic_torch_save", side_effect=interrupt):
            with self.assertRaises(InterruptedError):
                self.run_model(root, config, device)
        result, _ = self.run_model(root, config, device)
        baseline, resumed = self.load(baseline_root, config), self.load(root, config)
        for key in ("model_state_dict", "optimizer_state_dict", "rng_state", "shuffle_generator_state", "history", "best_checkpoints"):
            self.assert_nested_equal(baseline[key], resumed[key])
        for selection in experiment.CHECKPOINT_SELECTIONS:
            self.assertEqual(baseline_result["evaluations"][selection]["test_metrics"], result["evaluations"][selection]["test_metrics"])

    def test_interrupted_test_phase_resumes_without_more_training(self) -> None:
        root, config = self.case_root, self.config(epochs=2)
        original = experiment.evaluate_model
        calls = 0

        def interrupt(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise InterruptedError("second test evaluation")
            return original(*args, **kwargs)

        with mock.patch.object(experiment, "evaluate_model", side_effect=interrupt):
            with self.assertRaises(InterruptedError):
                self.run_model(root, config)
        result_path = root / "results" / "runs" / experiment.run_id_for(config) / "result.json"
        self.assertFalse(result_path.exists())
        with mock.patch.object(experiment, "run_epoch", side_effect=AssertionError("training already finished")):
            result, _ = self.run_model(root, config)
        self.assertEqual(set(result["evaluations"]), set(experiment.CHECKPOINT_SELECTIONS))
        preserved = {p: p.read_bytes() for p in (*self.paths(root, config).values(), result_path)}
        with mock.patch.object(experiment, "run_epoch", side_effect=AssertionError("unexpected training")), mock.patch.object(
            experiment, "evaluate_model", side_effect=AssertionError("unexpected evaluation"),
        ):
            _, skipped = self.run_model(root, config)
        self.assertTrue(skipped)
        for path, contents in preserved.items():
            self.assertEqual(path.read_bytes(), contents)

    def test_completed_run_restores_missing_bests_and_stale_status(self) -> None:
        root, config = self.case_root, self.config(epochs=1)
        result, _ = self.run_model(root, config)
        paths = self.paths(root, config)
        result_dir = root / "results" / "runs" / result["run_id"]
        preserved = {p: p.read_bytes() for p in (paths["latest"], result_dir / "result.json")}
        for selection in experiment.CHECKPOINT_SELECTIONS:
            paths[selection].unlink()
        experiment.save_status(result_dir / "status.json", status="interrupted", run_id=result["run_id"], error="after result commit")
        with mock.patch.object(experiment, "run_epoch", side_effect=AssertionError("unexpected training")), mock.patch.object(
            experiment, "evaluate_model", side_effect=AssertionError("unexpected evaluation"),
        ):
            _, skipped = self.run_model(root, config)
        self.assertTrue(skipped)
        latest = self.load(root, config)
        for selection in experiment.CHECKPOINT_SELECTIONS:
            self.assert_nested_equal(self.load(root, config, selection), latest["best_checkpoints"][selection])
        for path, contents in preserved.items():
            self.assertEqual(path.read_bytes(), contents)
        status = json.loads((result_dir / "status.json").read_text(encoding="utf-8"))
        self.assertEqual(status["status"], "completed")
        self.assertNotIn("error", status)

    def test_invalid_resume_and_partial_states_are_rejected(self) -> None:
        root, config = self.case_root, self.config(epochs=1)
        self.run_model(root, config)
        latest = self.load(root, config)
        for kind in ("tracker", "partial", "fingerprint"):
            with self.subTest(kind=kind):
                broken = copy.deepcopy(latest)
                if kind == "tracker":
                    broken["best_structure_epoch"] = 0
                elif kind == "partial":
                    broken["best_checkpoints"]["structure"]["model_state_dict"] = {
                        name: tensor for name, tensor in latest["model_state_dict"].items() if not name.startswith("global_sign_head.")}
                else:
                    broken["run_fingerprint"] = "another-run"
                with self.assertRaises(RuntimeError):
                    experiment.validate_resume_checkpoint(broken, config)
        paths = self.paths(root, config)
        paths["structure"].unlink()
        paths["latest"].unlink()
        with self.assertRaisesRegex(RuntimeError, "without latest.pt"):
            self.run_model(root, config)
        self.assertFalse(paths["structure"].exists())
        with self.assertRaisesRegex(RuntimeError, "baseline"):
            experiment.validate_resume_checkpoint({"model_state_dict": {}, "protocol_version": physics.PROTOCOL_VERSION}, config)

    def test_global_validation_does_not_change_structure_selection(self) -> None:
        config = self.config()
        state = experiment.copy_model_state(experiment.MODEL_REGISTRY[config["model"]["name"]].factory())
        selected = []
        for global_losses in ((0.0, 2.0, 3.0), (4.0, 3.0, 0.0)):
            best = {}
            for epoch, (structure, global_sign) in enumerate(zip((2.0, 1.0, 1.0), global_losses), start=1):
                snapshot = {name: torch.full_like(tensor, epoch) for name, tensor in state.items()}
                loss = physics.EpochLoss(structure + global_sign, structure, structure, 0.0, 0.0, global_sign)
                experiment.update_best_checkpoints(best, config=config, epoch=epoch, validation=loss, model_state=snapshot)
            selected.append(best)
        self.assertEqual([best["total"]["selected_epoch"] for best in selected], [1, 3])
        self.assertEqual([best["structure"]["selected_epoch"] for best in selected], [2, 2])
        self.assert_nested_equal(selected[0]["structure"]["model_state_dict"], selected[1]["structure"]["model_state_dict"])

    def fake_result(self, model_name: str, seed: int, fraction: float = 0.75, other_protocol: bool = False) -> dict:
        """Synthetic reporting arithmetic only; these are not experiment results."""
        config = self.config(model_name=model_name, seed=seed, fraction=fraction)
        if other_protocol:
            config["protocol_fingerprint"] = "other-protocol"
        evaluations = {}
        for selection, epoch in (("total", 1), ("structure", 2)):
            is_b = model_name == "g05_full_reconstruction"
            error = (12.0 if is_b else 10.0) if selection == "total" else (3.0 if is_b else 5.0)
            error += (seed - 41) * 0.1
            accuracy = (0.9 if is_b else 0.7) if selection == "total" else (0.75 if is_b else 0.8)
            metrics = {name: error if name in experiment.LOWER_IS_BETTER else accuracy for name in experiment.METRIC_NAMES}
            metrics.update(observed_sample_fraction=1.0, observations_per_sample=config["observation"]["g05_count_per_sample"])
            validation = physics.EpochLoss(2.5, 2.0, 2.0, 0.0, 0.0, 0.5)
            evaluations[selection] = {
                **config["training"]["checkpoint_selection"][selection], "selected_epoch": epoch,
                "selected_validation_loss": getattr(validation, selection), "validation_losses": asdict(validation),
                "test_metrics": metrics,
                "checkpoint_path": str(self.case_root / "fake" / experiment.run_id_for(config) / f"best_{selection}.pt"),
            }
        stopping = experiment.DualObjectiveEarlyStopping(experiment.regularization_from_config(config))
        for epoch, (total, structure) in enumerate(((2.5, 2.1), (2.6, 2.0), (2.6, 2.0)), 1):
            stopping.update(epoch, {"total": total, "structure": structure})
        return {"result_schema_version": experiment.RESULT_SCHEMA_VERSION, **experiment.run_metadata(config),
                "configuration": config, "status": "completed", "evaluations": evaluations,
                "training_result": {**experiment.completion_metadata(config, 3, stopping.state_dict()), "elapsed_seconds": 1.0,
                                    **experiment.best_tracking_fields(evaluations)}}

    def test_three_seed_reports_include_individual_pairs_and_correct_improvement(self) -> None:
        root = self.case_root
        for name in experiment.DEFAULT_MODELS:
            for seed in (41, 42, 43):
                result = self.fake_result(name, seed)
                experiment.atomic_write_json(root / "runs" / result["run_id"] / "result.json", result)
        fingerprint = result["protocol_fingerprint"]
        experiment.atomic_write_json(root / "runs" / "legacy" / "result.json", {"status": "completed", "protocol_fingerprint": "old"})
        self.assertTrue(experiment.refresh_reports(root, fingerprint))
        rows = {}
        for name in ("runs", "summary", "pairwise_comparisons", "pairwise_summary"):
            with (root / f"{name}.csv").open(encoding="utf-8", newline="") as handle:
                rows[name] = list(csv.DictReader(handle))
        self.assertEqual(len(rows["runs"]), 12)
        self.assertEqual(len(rows["summary"]), 4)
        self.assertEqual(len(rows["pairwise_comparisons"]), 2 * 3 * len(experiment.METRIC_NAMES))
        self.assertEqual(len(rows["pairwise_summary"]), 2 * len(experiment.METRIC_NAMES))
        for row in rows["pairwise_comparisons"]:
            self.assertEqual((row["model_a"], row["model_b"]), experiment.DEFAULT_MODELS)
            self.assertIn(row["seed"], ("41", "42", "43"))
            delta = float(row["value_b"]) - float(row["value_a"])
            self.assertAlmostEqual(float(row["delta_b_minus_a"]), delta)
            improvement = -delta if row["metric"] in experiment.LOWER_IS_BETTER else delta
            self.assertAlmostEqual(float(row["improvement_b_over_a"]), improvement)
        pairs = {(row["checkpoint_selection"], row["metric"]): row for row in rows["pairwise_summary"]}
        for selection, expected in (("total", -2.0), ("structure", 2.0)):
            for metric in ("mean_position_mae", "mean_position_3d_error", "charge_magnitude_mae"):
                row = pairs[(selection, metric)]
                self.assertEqual(row["paired_seed_count"], "3")
                self.assertEqual(row["paired_seeds"], "41,42,43")
                self.assertAlmostEqual(float(row["improvement_mean"]), expected)
                self.assertAlmostEqual(float(row["delta_std"]), 0.0)
        self.assertEqual(pairs[("structure", "global_sign_accuracy")]["metric_role"], "secondary")
        self.assertAlmostEqual(float(pairs[("structure", "relative_configuration_accuracy")]["improvement_mean"]), -0.05)
        for row in rows["summary"]:
            self.assertEqual(row["run_count"], "3")
            self.assertEqual(set(json.loads(row["selected_epochs_by_seed"])), {"41", "42", "43"})

    def test_reports_never_pair_different_protocols_selections_seeds_or_fractions(self) -> None:
        a = experiment.completed_result_evaluations(self.fake_result("g05_sign_only", 41))
        b = experiment.completed_result_evaluations(self.fake_result("g05_full_reconstruction", 41))
        self.assertEqual(experiment.build_pairwise_rows([a[0], b[1]]), [])
        for kwargs in (dict(seed=42), dict(seed=41, fraction=1.0), dict(seed=41, other_protocol=True)):
            other = experiment.completed_result_evaluations(self.fake_result("g05_full_reconstruction", **kwargs))
            self.assertEqual(experiment.build_pairwise_rows([*a, *other]), [])
        for builder in (experiment.build_pairwise_rows, experiment.build_summary_rows):
            with self.assertRaisesRegex(RuntimeError, "Duplicate"):
                builder([a[0], a[0]])
        for row in experiment.build_pairwise_summary_rows(experiment.build_pairwise_rows([*a, *b])):
            self.assertEqual(row["paired_seed_count"], 1)
            self.assertIsNone(row["delta_std"])
            self.assertIsNone(row["improvement_std"])

    def test_report_read_failures_preserve_all_csvs_and_removed_results_clear_rows(self) -> None:
        root = self.case_root
        result_paths = []
        for name in experiment.DEFAULT_MODELS:
            result = self.fake_result(name, 42)
            path = root / "runs" / result["run_id"] / "result.json"
            experiment.atomic_write_json(path, result)
            result_paths.append(path)
        fingerprint = result["protocol_fingerprint"]
        experiment.refresh_reports(root, fingerprint)
        files = [root / f"{name}.csv" for name in ("runs", "summary", "pairwise_comparisons", "pairwise_summary")]
        preserved = {path: path.read_bytes() for path in files}
        original_open = Path.open

        def unreadable(path, *args, **kwargs):
            if path == result_paths[1]:
                raise PermissionError("temporary Windows lock")
            return original_open(path, *args, **kwargs)

        with mock.patch.object(Path, "open", unreadable), contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(experiment.refresh_reports(root, fingerprint))
        for path, contents in preserved.items():
            self.assertEqual(path.read_bytes(), contents)
        for removed, counts in zip(reversed(result_paths), ((2, 2, 0, 0), (0, 0, 0, 0))):
            removed.unlink()
            experiment.refresh_reports(root, fingerprint)
            for path, count in zip(files, counts):
                with path.open(encoding="utf-8", newline="") as handle:
                    self.assertEqual(len(list(csv.DictReader(handle))), count)
                self.assertEqual(path.read_bytes().splitlines()[0], preserved[path].splitlines()[0])

    def test_zero_global_weight_metadata_reflects_actual_selection(self) -> None:
        weights = physics.LossWeights(global_sign=0.0)
        config = self.config(epochs=1, weights=weights)
        result, _ = self.run_model(self.case_root, config)
        for selection in experiment.CHECKPOINT_SELECTIONS:
            evaluation = result["evaluations"][selection]
            self.assertFalse(evaluation["global_sign_in_selection_objective"])
            self.assertGreater(evaluation["validation_losses"]["global_sign"], 0)
            self.assertEqual(evaluation["validation_losses"]["total"], evaluation["validation_losses"]["structure"])
            self.assertIsNotNone(evaluation["test_metrics"]["global_sign_accuracy"])
        for row in experiment.build_summary_rows(experiment.completed_result_evaluations(result)):
            self.assertFalse(row["global_sign_in_selection_objective"])
        self.assertIsNone(self.protocol()["training"]["checkpoint_selection"]["total"]["global_sign_in_selection_objective"])
        self.assertFalse(self.protocol(weights=weights)["training"]["checkpoint_selection"]["total"]["global_sign_in_selection_objective"])

    def test_protocol_guard_preserves_existing_files_and_tracks_both_sources(self) -> None:
        protocol = self.protocol()
        self.assertEqual(set(protocol["source_sha256"]), {"NewLearning9.py", "ModelExperiment10.py"})
        results, checkpoints = self.case_root / "results", self.case_root / "checkpoints"
        fingerprint = experiment.initialize_experiment_artifacts(experiment_results_dir=results, experiment_checkpoint_dir=checkpoints,
                                                                  protocol=protocol, split=self.split)
        preserved = {path: path.read_bytes() for path in results.iterdir() if path.is_file()}
        changed = copy.deepcopy(protocol)
        changed["source_sha256"]["NewLearning9.py"] = "changed-baseline"
        with self.assertRaisesRegex(RuntimeError, "new --experiment-name"):
            experiment.initialize_experiment_artifacts(experiment_results_dir=results, experiment_checkpoint_dir=checkpoints,
                                                      protocol=changed, split=self.split)
        self.assertEqual(fingerprint, experiment.object_fingerprint(protocol))
        for path, contents in preserved.items():
            self.assertEqual(path.read_bytes(), contents)
        self.assertEqual(experiment.initialize_experiment_artifacts(experiment_results_dir=results, experiment_checkpoint_dir=checkpoints,
                                                                   protocol=protocol, split=self.split), fingerprint)
        legacy = self.case_root / "legacy"
        experiment.atomic_write_json(legacy / "old_result.json", {"untouched": True})
        with self.assertRaisesRegex(RuntimeError, "Nonempty directory"):
            experiment.initialize_experiment_artifacts(experiment_results_dir=legacy, experiment_checkpoint_dir=self.case_root / "new",
                                                      protocol=protocol, split=self.split)
        self.assertEqual(json.loads((legacy / "old_result.json").read_text(encoding="utf-8")), {"untouched": True})

    def test_smoke_only_cli_checks_training_subset_without_creating_experiment(self) -> None:
        root = self.case_root
        args = ["--data", str(self.data_path), "--fractions", "0.75", "--seeds", "42", "--epochs", "300", "--device", "cpu",
                "--results-root", str(root / "results"), "--checkpoint-root", str(root / "checkpoints"),
                "--structure-dropout", "0.25", "--smoke-only"]
        with mock.patch.object(experiment, "train_and_evaluate_run", side_effect=AssertionError("unexpected full training")), mock.patch.object(
            experiment, "evaluate_model", side_effect=AssertionError("unexpected test-set evaluation"),
        ), contextlib.redirect_stdout(io.StringIO()) as output:
            experiment.main(args)
        self.assertIn("SMOKE PASS", output.getvalue())
        self.assertFalse((root / "results").exists())
        self.assertFalse((root / "checkpoints").exists())
        defaults = experiment.parse_args([])
        self.assertEqual(defaults.fractions, (0.75,))
        self.assertEqual(defaults.seeds, (42,))
        self.assertEqual(defaults.epochs, 300)
        self.assertEqual(defaults.models, experiment.DEFAULT_MODELS)
        self.assertEqual(defaults.batch_size, 128)
        self.assertEqual(defaults.data, experiment.DEFAULT_DATA_PATH)
        self.assertEqual(defaults.structure_dropout, 0.0)
        self.assertEqual(defaults.early_stopping_patience, 20)
        self.assertFalse(defaults.evaluate_only)
        self.assertTrue(experiment.parse_args(["--eval-only"]).evaluate_only)
        with contextlib.redirect_stderr(io.StringIO()):
            for invalid in (("--fractions", "nan"), ("--seeds", "42,42"), ("--models", "unknown"),
                            ("--experiment-name", "../escape"), ("--learning-rate", "inf"),
                            ("--smoke-only", "--evaluate-only"), ("--evaluate-only", "--batch-size", "0")):
                with self.assertRaises(SystemExit):
                    experiment.parse_args(list(invalid))

    def test_evaluate_only_cli_reuses_saved_protocol_without_updates_or_source_writes(self) -> None:
        weights = physics.LossWeights(position=1.7, global_sign=0.4)
        protocol = self.protocol(epochs=1, weights=weights)
        protocol["training"]["regularization"]["structure_dropout"] = 0.25
        protocol["source_sha256"]["ModelExperiment10.py"] = "0" * 64  # A saved earlier v10 source hash.
        protocol["environment"]["device"] = "cuda"  # Evaluation must permit a different device.
        results, checkpoints, args = self.evaluation_trial(protocol)
        originals = {}
        with contextlib.redirect_stdout(io.StringIO()):
            for name in experiment.DEFAULT_MODELS:
                config = experiment.run_configuration(protocol, model_name=name, fraction=0.75, seed=42)
                result, _ = experiment.train_and_evaluate_run(run_config=config, datasets=self.datasets[0.75],
                                                              experiment_results_dir=results, experiment_checkpoint_dir=checkpoints,
                                                              device=torch.device("cpu"))
                originals[result["run_id"]] = result
        preserved = {path: (path.read_bytes(), path.stat().st_mtime_ns)
                     for root in (results, checkpoints) for path in root.rglob("*") if path.is_file()}
        original_evaluate = experiment.evaluate_model

        def evaluate(model, dataset, stats, **kwargs):
            self.assertFalse(torch.is_grad_enabled())
            self.assertFalse(model.training)
            self.assertEqual(model.structure_dropout.p, 0.25)
            self.assertEqual(kwargs["batch_size"], 8)
            self.assertEqual(kwargs["weights"], weights)
            self.assertEqual(stats.to_dict(), self.stats.to_dict())
            self.assert_nested_equal(dataset.tensors, self.datasets[0.75][2].tensors)
            before = experiment.copy_model_state(model)
            metrics = original_evaluate(model, dataset, stats, **kwargs)
            self.assert_nested_equal(before, model.state_dict())
            self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
            return metrics

        with contextlib.ExitStack() as stack:
            for module, name in ((experiment, "train_and_evaluate_run"), (experiment, "run_epoch"),
                                 (experiment, "run_smoke_tests"), (experiment, "build_protocol"),
                                 (experiment, "atomic_torch_save"), (torch.optim, "AdamW"),
                                 (physics, "create_data_split"), (physics, "calculate_normalization_stats")):
                stack.enter_context(mock.patch.object(module, name, side_effect=AssertionError(f"unexpected {name}")))
            evaluated = stack.enter_context(mock.patch.object(experiment, "evaluate_model", side_effect=evaluate))
            prepared = stack.enter_context(mock.patch.object(physics, "prepare_dataset", wraps=physics.prepare_dataset))
            output = stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            # No --data or --batch-size: use the saved path/size, not training defaults.
            for _ in range(2):
                experiment.main([*args, "--epochs", "999", "--learning-rate", "0.2", "--weight-decay", "0.3",
                                 "--structure-dropout", "0.8", "--early-stopping-patience", "20",
                                 "--early-stopping-min-delta", "10", "--early-stopping-min-epochs", "999"])
            self.assertEqual(evaluated.call_count, 8)
            self.assertEqual(prepared.call_count, 2)
            for call in prepared.call_args_list:
                self.assert_nested_equal(call.args[1], self.split.test)
        self.assertIn("evaluated=2, failed=0", output.getvalue())
        evaluation_roots = list((results / "evaluations").iterdir())
        self.assertEqual(len(evaluation_roots), 2)  # A rerun neither skips evaluation nor overwrites it.
        for root in evaluation_roots:
            context = json.loads((root / "evaluation.json").read_text(encoding="utf-8"))
            self.assertEqual(context["environment"]["device"], "cpu")
            self.assertEqual(context["source_sha256"]["ModelExperiment10.py"], experiment.file_sha256(Path(experiment.__file__)))
            self.assertEqual(context["protocol_fingerprint"], experiment.object_fingerprint(protocol))
            for path in root.glob("runs/*/result.json"):
                result = json.loads(path.read_text(encoding="utf-8"))
                self.assertTrue(result["evaluation_only"])
                self.assertEqual(result["configuration"], originals[result["run_id"]]["configuration"])
                for selection in experiment.CHECKPOINT_SELECTIONS:
                    self.assertEqual(result["evaluations"][selection]["test_metrics"],
                                     originals[result["run_id"]]["evaluations"][selection]["test_metrics"])
            for name, expected in (("runs", 4), ("summary", 4), ("pairwise_comparisons", 30), ("pairwise_summary", 30)):
                with (root / f"{name}.csv").open(encoding="utf-8", newline="") as handle:
                    self.assertEqual(len(list(csv.DictReader(handle))), expected)
        for path, (contents, modified) in preserved.items():
            self.assertEqual(path.read_bytes(), contents)
            self.assertEqual(path.stat().st_mtime_ns, modified)

    def test_evaluate_only_uses_latest_snapshots_without_repairing_missing_files(self) -> None:
        root, config = self.case_root, self.config(epochs=1, fraction=0.0)
        original, _ = self.run_model(root, config)
        paths = self.paths(root, config)
        result_path = root / "results" / "runs" / original["run_id"] / "result.json"
        for path in (paths["total"], paths["structure"], result_path):
            path.unlink()
        before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
        with mock.patch.object(torch.optim, "AdamW", side_effect=AssertionError("unexpected optimizer")), mock.patch.object(
            experiment, "atomic_torch_save", side_effect=AssertionError("unexpected checkpoint repair"),
        ), contextlib.redirect_stdout(io.StringIO()):
            result = experiment.evaluate_only_run(run_config=config, test=self.datasets[0.0][2],
                                                   experiment_results_dir=root / "results", experiment_checkpoint_dir=root / "checkpoints",
                                                   device=torch.device("cpu"), batch_size=1)
        self.assertEqual(result["evaluation_batch_size"], 1)
        for selection in experiment.CHECKPOINT_SELECTIONS:
            self.assertFalse(paths[selection].exists())
            evaluation = result["evaluations"][selection]
            self.assertEqual(evaluation["checkpoint_path"], str(paths["latest"].resolve()))
            self.assertEqual(evaluation["checkpoint_source"], f"best_checkpoints.{selection}")
            self.assertIsNone(evaluation["test_metrics"]["global_sign_accuracy"])
            for metric, expected in original["evaluations"][selection]["test_metrics"].items():
                if expected is not None:
                    self.assertAlmostEqual(evaluation["test_metrics"][metric], expected, places=6)
        self.assertFalse(result_path.exists())
        self.assertEqual({path: path.read_bytes() for path in root.rglob("*") if path.is_file()}, before)

    def test_evaluate_only_can_use_selected_files_with_completed_result_without_latest(self) -> None:
        root, config = self.case_root, self.config(epochs=1)
        original, _ = self.run_model(root, config)
        paths = self.paths(root, config)
        paths["latest"].unlink()
        with mock.patch.object(torch.optim, "AdamW", side_effect=AssertionError("unexpected optimizer")), contextlib.redirect_stdout(io.StringIO()):
            result = experiment.evaluate_only_run(run_config=config, test=self.datasets[0.75][2],
                                                   experiment_results_dir=root / "results", experiment_checkpoint_dir=root / "checkpoints",
                                                   device=torch.device("cpu"))
        for selection in experiment.CHECKPOINT_SELECTIONS:
            self.assertEqual(result["evaluations"][selection]["test_metrics"], original["evaluations"][selection]["test_metrics"])
        self.assertFalse(paths["latest"].exists())
        paths["structure"].unlink()
        with mock.patch.object(experiment, "evaluate_model", side_effect=AssertionError("unexpected partial evaluation")):
            with self.assertRaisesRegex(FileNotFoundError, "Missing evaluation checkpoint"):
                experiment.evaluate_only_run(run_config=config, test=self.datasets[0.75][2],
                                             experiment_results_dir=root / "results", experiment_checkpoint_dir=root / "checkpoints",
                                             device=torch.device("cpu"))

    def test_evaluate_only_missing_or_unfinished_training_never_starts_training(self) -> None:
        protocol = self.protocol(epochs=2)
        results, checkpoints, args = self.evaluation_trial(protocol)
        args.extend(["--models", "g05_full_reconstruction"])
        with mock.patch.object(experiment, "train_and_evaluate_run", side_effect=AssertionError("unexpected training")), mock.patch.object(
            experiment, "run_smoke_tests", side_effect=AssertionError("unexpected smoke test"),
        ), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(FileNotFoundError, "No completed training"):
                experiment.main(args)
        self.assertEqual(list(checkpoints.iterdir()), [])
        config = experiment.run_configuration(protocol, model_name="g05_full_reconstruction", fraction=0.75, seed=42)
        original_save = experiment.atomic_torch_save

        def interrupt(value, path):
            original_save(value, path)
            if path.name == "latest.pt" and value["epoch"] == 1:
                raise InterruptedError("stop after first epoch")

        with mock.patch.object(experiment, "atomic_torch_save", side_effect=interrupt), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(InterruptedError):
                experiment.train_and_evaluate_run(run_config=config, datasets=self.datasets[0.75],
                                                 experiment_results_dir=results, experiment_checkpoint_dir=checkpoints,
                                                 device=torch.device("cpu"))
        before = {path: path.read_bytes() for path in checkpoints.rglob("*.pt")}
        with contextlib.ExitStack() as stack:
            for name in ("train_and_evaluate_run", "run_smoke_tests", "run_epoch", "evaluate_model"):
                stack.enter_context(mock.patch.object(experiment, name, side_effect=AssertionError(f"unexpected {name}")))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            with self.assertRaisesRegex(RuntimeError, "unfinished training"):
                experiment.main(args)
        self.assertEqual({path: path.read_bytes() for path in checkpoints.rglob("*.pt")}, before)

    def test_evaluate_only_rejects_changed_data_protocol_and_split(self) -> None:
        protocol = self.protocol(epochs=1)
        results, _, args = self.evaluation_trial(protocol)
        protocol_path = results / "protocol.json"
        saved = json.loads(protocol_path.read_text(encoding="utf-8"))
        invalid = copy.deepcopy(saved)
        invalid["data"]["sample_count"] += 1
        experiment.atomic_write_json(protocol_path, invalid)
        with self.assertRaisesRegex(RuntimeError, "fingerprint"):
            experiment.main(args)
        experiment.atomic_write_json(protocol_path, saved)
        # A relocated byte-identical copy is accepted; unrelated data are not.
        relocated = self.case_root / "relocated.npz"
        relocated.write_bytes(self.data_path.read_bytes())
        restored, _, split, path = experiment.load_evaluation_data(results, relocated)
        self.assertEqual(restored, protocol)
        self.assertEqual(path, relocated.resolve())
        self.assert_nested_equal(split.test, self.split.test)
        relocated.write_bytes(b"not the training dataset")
        with self.assertRaisesRegex(RuntimeError, "SHA256"):
            experiment.main([*args, "--data", str(relocated)])
        experiment.atomic_save_npz(results / "split_indices.npz", train=self.split.train,
                                   validation=self.split.validation, test=self.split.test[::-1])
        with self.assertRaisesRegex(RuntimeError, "Saved split"):
            experiment.main(args)
        experiment.atomic_save_npz(results / "split_indices.npz", train=self.split.train,
                                   validation=self.split.validation, test=self.split.test)
        invalid_normalization = {**protocol["normalization"], "g00_mean": 999}
        experiment.atomic_write_json(results / "normalization.json", invalid_normalization)
        with self.assertRaisesRegex(RuntimeError, "normalization"):
            experiment.main(args)
        self.assertFalse((results / "evaluations").exists())

    def test_evaluate_only_continue_on_error_keeps_successes_separate_from_training(self) -> None:
        protocol = self.protocol(epochs=1)
        results, checkpoints, args = self.evaluation_trial(protocol)
        config = experiment.run_configuration(protocol, model_name="g05_full_reconstruction", fraction=0.75, seed=42)
        with contextlib.redirect_stdout(io.StringIO()):
            experiment.train_and_evaluate_run(run_config=config, datasets=self.datasets[0.75],
                                             experiment_results_dir=results, experiment_checkpoint_dir=checkpoints,
                                             device=torch.device("cpu"))
        original_dirs = list(checkpoints.iterdir())
        with mock.patch.object(experiment, "train_and_evaluate_run", side_effect=AssertionError("unexpected training")), mock.patch.object(
            experiment, "run_smoke_tests", side_effect=AssertionError("unexpected smoke test"),
        ), contextlib.redirect_stdout(io.StringIO()) as output:
            with self.assertRaisesRegex(RuntimeError, "1 evaluations failed"):
                experiment.main([*args, "--models", "g05_full_reconstruction", "--seeds", "43,42", "--continue-on-error"])
        self.assertIn("evaluated=1, failed=1", output.getvalue())
        self.assertEqual(list(checkpoints.iterdir()), original_dirs)
        evaluation_root, = (results / "evaluations").iterdir()
        self.assertEqual(len(list(evaluation_root.glob("runs/*/result.json"))), 1)
        self.assertEqual(len(list(evaluation_root.glob("runs/*/status.json"))), 1)
        with (evaluation_root / "runs.csv").open(encoding="utf-8", newline="") as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 2)

    def test_inference_loader_rejects_baseline_checkpoint_with_clear_error(self) -> None:
        path = self.case_root / "legacy.pt"
        experiment.atomic_torch_save({"protocol_version": physics.PROTOCOL_VERSION, "charge_count": 5,
                                      "model_state_dict": {}}, path)
        with self.assertRaisesRegex(RuntimeError, "routing experiment"):
            experiment.load_trained_model(path)

    def test_resume_rejects_nonfinite_training_history(self) -> None:
        root, config = self.case_root, self.config(epochs=1)
        self.run_model(root, config)
        latest = self.load(root, config)
        latest["history"][0]["train"]["position"] = float("nan")
        with self.assertRaisesRegex(RuntimeError, "history"):
            experiment.validate_resume_checkpoint(latest, config)

    def test_results_reject_missing_required_metrics_and_wrong_observation_counts(self) -> None:
        for metric in ("mean_position_mae", "global_sign_accuracy", "observations_per_sample"):
            with self.subTest(metric=metric):
                result = self.fake_result("g05_full_reconstruction", 42)
                result["evaluations"]["structure"]["test_metrics"][metric] = None
                with self.assertRaisesRegex(RuntimeError, "metrics"):
                    experiment.completed_result_evaluations(result)
        result = self.fake_result("g05_full_reconstruction", 42)
        result["evaluations"]["structure"]["test_metrics"]["observations_per_sample"] = 32
        with self.assertRaisesRegex(RuntimeError, "observation"):
            experiment.completed_result_evaluations(result)

    def test_concurrent_experiment_locks_reject_writers_and_release_after_error(self) -> None:
        roots = (self.case_root / "results" / "trial", self.case_root / "checkpoints" / "trial")
        with self.assertRaisesRegex(InterruptedError, "simulated stop"):
            with experiment.experiment_locks(*roots):
                for root in roots:
                    with self.assertRaisesRegex(RuntimeError, "already running"):
                        with experiment.experiment_locks(root):
                            self.fail("Concurrent writer entered")
                raise InterruptedError("simulated stop")
        with experiment.experiment_locks(*roots):
            pass  # Existing lock files must not block a subsequent resume.
        for root in roots:
            self.assertFalse(root.exists())
            self.assertTrue(root.with_name(f".{root.name}.lock").is_file())

    def test_training_rejects_a_mask_that_disagrees_with_run_configuration(self) -> None:
        config = self.config(epochs=1)
        original = self.datasets[0.75]
        mask = original[0].tensors[2].clone()
        mask[:, 0] = 0
        bad = TensorDataset(*original[0].tensors[:2], mask, *original[0].tensors[3:])
        with self.assertRaisesRegex(ValueError, "fixed G05 prefix"):
            experiment.train_and_evaluate_run(run_config=config, datasets=(bad, *original[1:]),
                                             experiment_results_dir=self.case_root / "results",
                                             experiment_checkpoint_dir=self.case_root / "checkpoints", device=torch.device("cpu"))
        self.assertFalse(self.paths(self.case_root, config)["latest"].exists())

    def test_cli_training_and_completed_rerun_preserve_checkpoint_and_report_contents(self) -> None:
        root = self.case_root
        args = ["--data", str(self.data_path), "--fractions", "0.75", "--seeds", "42", "--epochs", "2", "--batch-size", "8",
                "--device", "cpu", "--experiment-name", "trial", "--results-root", str(root / "results"),
                "--checkpoint-root", str(root / "checkpoints")]
        with contextlib.redirect_stdout(io.StringIO()):
            experiment.main(args)
        reports = root / "results" / "trial"
        checkpoints = root / "checkpoints" / "trial"
        self.assertEqual(len(list(reports.glob("runs/*/result.json"))), 2)
        for name, expected in (("runs", 4), ("summary", 4), ("pairwise_comparisons", 30), ("pairwise_summary", 30)):
            with (reports / f"{name}.csv").open(encoding="utf-8", newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), expected)
        kept = [*checkpoints.glob("*/*.pt"), *reports.glob("*.csv"), *reports.glob("runs/*/result.json")]
        before = {path: path.read_bytes() for path in kept}
        with mock.patch.object(experiment, "run_epoch", side_effect=AssertionError("unexpected training")), mock.patch.object(
            experiment, "evaluate_model", side_effect=AssertionError("unexpected test evaluation"),
        ), contextlib.redirect_stdout(io.StringIO()) as output:
            experiment.main(args)
        self.assertIn("skipped=2", output.getvalue())
        for path, contents in before.items():
            self.assertEqual(path.read_bytes(), contents)


    def regularized_config(self, *, model_name: str = "g05_full_reconstruction", fraction: float = 0.75,
                           epochs: int = 12, dropout: float = 0.25, patience: int = 2,
                           min_delta: float = 0.0, min_epochs: int = 0,
                           device: torch.device = torch.device("cpu")) -> dict:
        protocol = self.protocol(epochs=epochs, device=device)
        protocol["training"]["regularization"] = asdict(experiment.RegularizationSettings(
            dropout, patience, min_delta, min_epochs))
        return experiment.run_configuration(protocol, model_name=model_name, fraction=fraction, seed=42)

    def plateau_epoch(self, model, loader, optimizer=None, weights=physics.LossWeights()):
        """실제 학습·드롭아웃·optimizer를 실행하되 종료 시나리오용 검증 손실만 고정한다."""
        actual = physics.run_epoch(model, loader, optimizer, weights)
        if optimizer is not None:
            return actual
        observed = bool(torch.any(loader.dataset.tensors[2]))
        return physics.EpochLoss(1.5 if observed else 1.0, 1.0, 0.7, 0.2, 0.1, 0.5 if observed else None)

    def test_disabled_controls_match_both_v9_routes_epoch_for_epoch(self) -> None:
        # NewLearning9 sign-only뿐 아니라 v9의 full G05 경로도 원형과 비교한다.
        for name in experiment.DEFAULT_MODELS:
            for fraction in (0.0, 0.75):
                with self.subTest(model=name, fraction=fraction):
                    experiment.set_reproducibility(42)
                    old = previous_experiment.MODEL_REGISTRY[name].factory()
                    experiment.set_reproducibility(42)
                    new = experiment.model_from_config(self.config(model_name=name, fraction=fraction))
                    self.assert_nested_equal(old.state_dict(), new.state_dict())
                    optimizers = [torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4) for m in (old, new)]
                    loaders = [physics.create_data_loader(self.datasets[fraction][0], 8, shuffle=True,
                                                         seed=42, device=torch.device("cpu")) for _ in (old, new)]
                    for _ in range(3):
                        expected = previous_experiment.run_epoch(old, loaders[0], optimizers[0])
                        actual = experiment.run_epoch(new, loaders[1], optimizers[1])
                        self.assertEqual(asdict(expected), asdict(actual))
                        self.assert_nested_equal(old.state_dict(), new.state_dict())
                        self.assert_nested_equal(optimizers[0].state_dict(), optimizers[1].state_dict())

    def test_dual_stopping_resets_when_either_objective_improves(self) -> None:
        tracker = experiment.DualObjectiveEarlyStopping(experiment.RegularizationSettings(early_stopping_patience=2))
        for epoch, (structure, total) in enumerate(((3, 5), (2, 6), (3, 4), (3, 5), (3, 5)), 1):
            stopped = tracker.update(epoch, {"structure": structure, "total": total})
            self.assertEqual(stopped, epoch == 5)
        self.assertEqual(tracker.state_dict(), {"epoch": 5, "best_losses": {"total": 4.0, "structure": 2.0},
                                               "last_improvement_epoch": 3, "bad_epochs": 2, "stopped": True})
        with self.assertRaisesRegex(RuntimeError, "cannot continue"):
            tracker.update(6, {"structure": 0.0, "total": 0.0})

    def test_min_delta_accumulates_and_does_not_change_raw_best_selection(self) -> None:
        config = self.regularized_config(dropout=0, min_delta=0.25, patience=2)
        tracker = experiment.DualObjectiveEarlyStopping(experiment.regularization_from_config(config))
        state = experiment.copy_model_state(experiment.model_from_config(config))
        best = {}
        # 정확히 표현 가능한 1/8 단위로 strict '<' 경계도 검사한다.
        for epoch, structure in enumerate((2.0, 1.875, 1.75), 1):
            validation = physics.EpochLoss(structure + 0.5, structure, structure, 0.0, 0.0, 0.5)
            tracker.update(epoch, validation)
            experiment.update_best_checkpoints(best, config=config, epoch=epoch, validation=validation, model_state=state)
        self.assertTrue(tracker.stopped)
        self.assertEqual(tracker.best_losses, {"total": 2.5, "structure": 2.0})
        self.assertEqual({k: v["selected_epoch"] for k, v in best.items()}, {"total": 3, "structure": 3})
        cumulative = experiment.DualObjectiveEarlyStopping(experiment.RegularizationSettings(
            early_stopping_patience=3, early_stopping_min_delta=0.25))
        for epoch, score in enumerate((2.0, 1.875, 1.75, 1.625), 1):
            cumulative.update(epoch, {"total": score, "structure": score})
        self.assertFalse(cumulative.stopped)
        self.assertEqual(cumulative.bad_epochs, 0)
        self.assertEqual(cumulative.last_improvement_epoch, 4)

    def test_stopping_warmup_disabled_and_invalid_inputs(self) -> None:
        for patience, minimum, expected_epoch in ((2, 5, 5), (0, 0, None)):
            tracker = experiment.DualObjectiveEarlyStopping(experiment.RegularizationSettings(
                early_stopping_patience=patience, early_stopping_min_epochs=minimum))
            for epoch in range(1, (expected_epoch or 12) + 1):
                self.assertEqual(tracker.update(epoch, {"total": 1.0, "structure": 1.0}), epoch == expected_epoch)
        with contextlib.redirect_stderr(io.StringIO()):
            for flag, value in (("--structure-dropout", "1"), ("--structure-dropout", "nan"),
                                ("--structure-dropout", "-0.1"), ("--early-stopping-patience", "-1"),
                                ("--early-stopping-min-delta", "inf"), ("--early-stopping-min-delta", "-0.01"),
                                ("--early-stopping-min-epochs", "-2")):
                with self.subTest(flag=flag, value=value), self.assertRaises(SystemExit):
                    experiment.parse_args([flag, value])
        tracker = experiment.DualObjectiveEarlyStopping(experiment.RegularizationSettings())
        before = tracker.state_dict()
        with self.assertRaises(FloatingPointError):
            tracker.update(1, {"total": 1.0, "structure": float("nan")})
        self.assertEqual(tracker.state_dict(), before)
        with self.assertRaisesRegex(RuntimeError, "consecutive"):
            tracker.update(2, {"total": 1.0, "structure": 1.0})

    def test_regularization_settings_change_identity_and_survive_inference_load(self) -> None:
        base = self.regularized_config(epochs=1)
        for changes in ({"dropout": 0.3}, {"patience": 4}, {"min_delta": 0.01}, {"min_epochs": 6}):
            self.assertNotEqual(experiment.run_id_for(base), experiment.run_id_for(self.regularized_config(epochs=1, **changes)))
        self.run_model(self.case_root, base)
        loaded, _, checkpoint = experiment.load_trained_model(self.paths(self.case_root, base)["structure"])
        self.assertFalse(loaded.training)
        self.assertEqual(loaded.structure_dropout.p, 0.25)
        self.assertEqual(checkpoint["configuration"]["training"]["regularization"], base["training"]["regularization"])
        path = self.case_root / "v9.pt"
        experiment.atomic_torch_save({"protocol_version": previous_experiment.PROTOCOL_VERSION,
                                     "checkpoint_schema_version": previous_experiment.CHECKPOINT_SCHEMA_VERSION}, path)
        with self.assertRaisesRegex(RuntimeError, "not a checkpoint"):
            experiment.load_trained_model(path)


    def test_dropout_is_training_only_and_preserves_symmetry_and_gradient_routes(self) -> None:
        batch = tuple(t[:6] for t in self.datasets[0.75][0].tensors)
        g00, g05, mask, position, charge = batch
        reversed_g05 = g05 * g05.new_tensor((1, 1, -1))
        for name in experiment.DEFAULT_MODELS:
            model = experiment.model_from_config(self.regularized_config(model_name=name, dropout=0.4))
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            # context가 실제 활성화된 뒤에도 성질이 유지되는지 확인한다.
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                physics.calculate_losses(model(*batch[:3]), position, charge, mask).total.backward()
                optimizer.step()
            model.train()
            rng = experiment.capture_rng_state()
            output = model(*batch[:3])
            experiment.restore_rng_state(rng)
            flipped = model(g00, reversed_g05, mask)
            for field in experiment.OUTPUT_FIELDS[:-1]:
                torch.testing.assert_close(getattr(output, field), getattr(flipped, field), rtol=1e-6, atol=1e-7)
            torch.testing.assert_close(output.global_sign_logit, -flipped.global_sign_logit, rtol=1e-6, atol=1e-7)
            independent = model(*batch[:3])
            self.assertFalse(torch.equal(output.position, independent.position))
            torch.testing.assert_close(output.global_sign_logit, independent.global_sign_logit, rtol=0, atol=0)
            optimizer.zero_grad(set_to_none=True)
            physics.calculate_losses(model(*batch[:3]), position, charge, mask).global_sign.backward()
            experiment.assert_no_gradient(model, experiment.STRUCTURE_PREFIXES)
            optimizer.zero_grad(set_to_none=True)
            physics.calculate_losses(model(*batch[:3]), position, charge, mask).structure.backward()
            experiment.assert_no_gradient(model, ("global_sign_head.",))
            if name == "g05_sign_only":
                experiment.assert_no_gradient(model, ("g05_encoder.", "structure_context."))
            else:
                self.assertTrue(experiment.has_nonzero_gradient(model, ("g05_encoder.", "structure_context.")))
            model.eval()
            with torch.no_grad():
                expected = model(*batch[:3])
                experiment.assert_outputs_close(expected, model(*batch[:3]))
                hidden = g05.masked_fill(~mask.bool().expand_as(g05), float("nan"))
                experiment.assert_outputs_close(expected, model(g00, hidden, mask))
                zero = model(g00, torch.full_like(g05, float("nan")), torch.zeros_like(mask))
                self.assertTrue(torch.equal(zero.global_sign_logit, torch.zeros_like(zero.global_sign_logit)))

    def test_dropout_sign_only_structure_stays_independent_of_g05_fraction(self) -> None:
        references = None
        for fraction in (0.0, 0.75, 1.0):
            config = self.regularized_config(model_name="g05_sign_only", fraction=fraction, epochs=3, patience=0)
            root = self.case_root / str(fraction)
            self.run_model(root, config)
            latest = self.load(root, config)
            structure = {k: v for k, v in latest["model_state_dict"].items() if k.startswith(experiment.STRUCTURE_PREFIXES)}
            losses = [h["train"]["structure"] for h in latest["history"]]
            if references is None:
                references = structure, losses
            self.assert_nested_equal(structure, references[0])
            self.assertEqual(losses, references[1])

    def test_dropout_zero_g05_both_models_still_match_and_stop(self) -> None:
        checkpoints = []
        for name in experiment.DEFAULT_MODELS:
            config = self.regularized_config(model_name=name, fraction=0.0)
            root = self.case_root / name
            with mock.patch.object(experiment, "run_epoch", side_effect=self.plateau_epoch):
                result, _ = self.run_model(root, config)
            self.assertEqual(result["training_result"]["epochs_completed"], 3)
            self.assertEqual(result["training_result"]["stop_reason"], "early_stopping")
            for evaluation in result["evaluations"].values():
                self.assertIsNone(evaluation["test_metrics"]["global_sign_accuracy"])
                self.assertIsNone(evaluation["validation_losses"]["global_sign"])
            checkpoints.append(self.load(root, config))
        for key in ("model_state_dict", "optimizer_state_dict", "rng_state", "shuffle_generator_state", "history", "early_stopping"):
            self.assert_nested_equal(checkpoints[0][key], checkpoints[1][key])

    def check_regularized_resume(self, device: torch.device) -> None:
        config = self.regularized_config(device=device)
        base_root = self.case_root / "baseline"
        with mock.patch.object(experiment, "run_epoch", side_effect=self.plateau_epoch):
            baseline_result, _ = self.run_model(base_root, config, device)
        baseline = self.load(base_root, config)
        self.assertEqual(baseline["epoch"], 3)
        for epoch, boundary in ((2, "before"), (2, "after"), (3, "before"), (3, "after")):
            with self.subTest(device=str(device), epoch=epoch, boundary=boundary):
                root = self.case_root / f"epoch{epoch}_{boundary}"
                original = experiment.atomic_torch_save

                def interrupt(value, path):
                    target = path.name == "latest.pt" and value["epoch"] == epoch
                    if target and boundary == "before":
                        raise InterruptedError("before commit")
                    original(value, path)
                    if target and boundary == "after":
                        raise InterruptedError("after commit")

                with mock.patch.object(experiment, "atomic_torch_save", side_effect=interrupt), mock.patch.object(
                    experiment, "run_epoch", side_effect=self.plateau_epoch,
                ), self.assertRaises(InterruptedError):
                    self.run_model(root, config, device)
                # 종료 스냅샷을 저장했다면 학습/검증 epoch 호출 자체가 없어야 한다.
                side_effect = AssertionError("terminal checkpoint must not train") if (epoch, boundary) == (3, "after") else self.plateau_epoch
                with mock.patch.object(experiment, "run_epoch", side_effect=side_effect):
                    result, _ = self.run_model(root, config, device)
                resumed = self.load(root, config)
                for key in ("model_state_dict", "optimizer_state_dict", "rng_state", "shuffle_generator_state",
                            "history", "best_checkpoints", "early_stopping"):
                    self.assert_nested_equal(baseline[key], resumed[key])
                self.assertEqual(result["training_result"]["epochs_completed"], 3)
                for selection in experiment.CHECKPOINT_SELECTIONS:
                    self.assertEqual(baseline_result["evaluations"][selection]["test_metrics"], result["evaluations"][selection]["test_metrics"])

    def test_dropout_cpu_resume_is_exact_before_and_after_stop_commit(self) -> None:
        self.check_regularized_resume(torch.device("cpu"))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_dropout_cuda_resume_is_exact_before_and_after_stop_commit(self) -> None:
        self.check_regularized_resume(torch.device("cuda"))


    def test_early_stop_evaluation_interruption_repair_and_read_only_evaluation(self) -> None:
        root, config = self.case_root, self.regularized_config()
        with mock.patch.object(experiment, "run_epoch", side_effect=self.plateau_epoch), mock.patch.object(
            experiment, "evaluate_model", side_effect=InterruptedError("evaluation interrupted after stopping"),
        ), self.assertRaises(InterruptedError):
            self.run_model(root, config)
        latest = self.load(root, config)
        self.assertEqual(latest["epoch"], 3)
        self.assertTrue(latest["early_stopping"]["stopped"])
        with mock.patch.object(experiment, "run_epoch", side_effect=AssertionError("already stopped")):
            result, _ = self.run_model(root, config)
        self.assertTrue(experiment.refresh_reports(root / "results", config["protocol_fingerprint"]))
        with (root / "results" / "runs.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertEqual(row["stop_reason"], "early_stopping")
            self.assertEqual(row["epochs_completed"], "3")
            self.assertEqual(row["structure_dropout"], "0.25")
        paths = self.paths(root, config)
        # 완료된 조기 종료 실행도 latest로 유실된 두 선택 파일을 복구할 수 있다.
        for selection in experiment.CHECKPOINT_SELECTIONS:
            paths[selection].unlink()
        with mock.patch.object(experiment, "run_epoch", side_effect=AssertionError("already stopped")), mock.patch.object(
            experiment, "evaluate_model", side_effect=AssertionError("already evaluated"),
        ):
            _, skipped = self.run_model(root, config)
        self.assertTrue(skipped)
        for selection in experiment.CHECKPOINT_SELECTIONS:
            self.assert_nested_equal(self.load(root, config, selection), latest["best_checkpoints"][selection])
        # evaluation-only는 누락 파일을 디스크에 복구하지 않고 latest 안에서 읽는다.
        for selection in experiment.CHECKPOINT_SELECTIONS:
            paths[selection].unlink()
        before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
        with mock.patch.object(experiment, "run_epoch", side_effect=AssertionError("evaluation only")), mock.patch.object(
            torch.optim, "AdamW", side_effect=AssertionError("no optimizer"),
        ), contextlib.redirect_stdout(io.StringIO()):
            evaluation = experiment.evaluate_only_run(run_config=config, test=self.datasets[0.75][2],
                                                       experiment_results_dir=root / "results",
                                                       experiment_checkpoint_dir=root / "checkpoints", device=torch.device("cpu"))
        self.assertEqual(evaluation["training_result"]["stop_reason"], "early_stopping")
        for selection in experiment.CHECKPOINT_SELECTIONS:
            self.assertEqual(evaluation["evaluations"][selection]["test_metrics"], result["evaluations"][selection]["test_metrics"])
        self.assertEqual({p: p.read_bytes() for p in root.rglob("*") if p.is_file()}, before)
        # latest 없이도 완료 결과와 두 best만 있으면 기존 평가 경로를 사용할 수 있다.
        for selection in experiment.CHECKPOINT_SELECTIONS:
            experiment.atomic_torch_save(latest["best_checkpoints"][selection], paths[selection])
        paths["latest"].unlink()
        with contextlib.redirect_stdout(io.StringIO()):
            selected_only = experiment.evaluate_only_run(run_config=config, test=self.datasets[0.75][2],
                                                         experiment_results_dir=root / "results",
                                                         experiment_checkpoint_dir=root / "checkpoints", device=torch.device("cpu"))
        self.assertEqual(selected_only["training_result"]["stop_reason"], "early_stopping")

    def test_saved_early_stop_state_and_completion_cannot_be_forged(self) -> None:
        config = self.regularized_config()
        with mock.patch.object(experiment, "run_epoch", side_effect=self.plateau_epoch):
            result, _ = self.run_model(self.case_root, config)
        latest = self.load(self.case_root, config)
        for key, value in (("bad_epochs", 0), ("stopped", False), ("last_improvement_epoch", 2)):
            altered = copy.deepcopy(latest)
            altered["early_stopping"][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, "Early stopping state"):
                experiment.validate_resume_checkpoint(altered, config)
        altered = copy.deepcopy(latest)
        altered["epoch"] += 1
        altered["history"].append({**altered["history"][-1], "epoch": altered["epoch"]})
        with self.assertRaisesRegex(RuntimeError, "cannot continue"):
            experiment.validate_resume_checkpoint(altered, config)
        for mutation in ("reason", "epochs", "missing_state", "premature"):
            altered = copy.deepcopy(result)
            if mutation == "reason":
                altered["training_result"]["stop_reason"] = "max_epochs"
            elif mutation == "epochs":
                altered["training_result"]["epochs_completed"] = config["training"]["max_epochs"]
            elif mutation == "missing_state":
                del altered["training_result"]["early_stopping"]
            else:
                altered["training_result"]["epochs_completed"] = 2
                altered["training_result"]["early_stopping"].update(epoch=2, bad_epochs=1, stopped=False)
            with self.subTest(mutation=mutation), self.assertRaises(RuntimeError):
                experiment.completed_result_evaluations(altered)


if __name__ == "__main__":
    unittest.main()
