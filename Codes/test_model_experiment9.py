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

import ModelExperiment9 as experiment
import NewLearning9 as physics
import generate_charge_dataset as generator


class FiveChargeExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, previous_threads)
        cls.temporary = tempfile.TemporaryDirectory(prefix="m9-tests-")
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
        return {"result_schema_version": experiment.RESULT_SCHEMA_VERSION, **experiment.run_metadata(config),
                "configuration": config, "status": "completed", "evaluations": evaluations,
                "training_result": {"epochs_completed": 3, "elapsed_seconds": 1.0,
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
        self.assertEqual(set(protocol["source_sha256"]), {"NewLearning9.py", "ModelExperiment9.py"})
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
                "--results-root", str(root / "results"), "--checkpoint-root", str(root / "checkpoints"), "--smoke-only"]
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
        self.assertFalse(defaults.evaluate_only)
        self.assertTrue(experiment.parse_args(["--eval-only"]).evaluate_only)
        with contextlib.redirect_stderr(io.StringIO()):
            for invalid in (("--fractions", "nan"), ("--seeds", "42,42"), ("--models", "unknown"),
                            ("--experiment-name", "../escape"), ("--learning-rate", "inf"),
                            ("--smoke-only", "--evaluate-only"), ("--evaluate-only", "--batch-size", "0"),
                            ("--evaluation-results-dir", "saved"), ("--evaluation-checkpoint-dir", "saved")):
                with self.assertRaises(SystemExit):
                    experiment.parse_args(list(invalid))

    def test_evaluate_only_cli_reuses_saved_protocol_without_updates_or_source_writes(self) -> None:
        weights = physics.LossWeights(position=1.7, global_sign=0.4)
        protocol = self.protocol(epochs=1, weights=weights)
        protocol["source_sha256"]["ModelExperiment9.py"] = "0" * 64  # Before evaluation-only support.
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
                experiment.main([*args, "--epochs", "999", "--learning-rate", "0.2", "--weight-decay", "0.3"])
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
            self.assertEqual(context["source_sha256"]["ModelExperiment9.py"], experiment.file_sha256(Path(experiment.__file__)))
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

    def test_evaluate_only_accepts_moved_seed_results_and_documented_legacy_source(self) -> None:
        protocol = self.protocol(epochs=1)
        protocol.pop("source_ast_sha256")  # Original v1 protocols stored only the raw source hash.
        protocol["source_sha256"]["NewLearning9.py"] = "4768dd7dd514605d62642c39943d4a0655dc8058ccf8414c3efc1600f5df16cd"
        results = self.case_root / "results" / "trial_seed42"
        checkpoints = self.case_root / "checkpoints" / "trial"
        experiment.initialize_experiment_artifacts(experiment_results_dir=results, experiment_checkpoint_dir=checkpoints,
                                                  protocol=protocol, split=self.split)
        config = experiment.run_configuration(protocol, model_name="g05_full_reconstruction", fraction=0.75, seed=42)
        with contextlib.redirect_stdout(io.StringIO()):
            original, _ = experiment.train_and_evaluate_run(run_config=config, datasets=self.datasets[0.75],
                                                           experiment_results_dir=results, experiment_checkpoint_dir=checkpoints,
                                                           device=torch.device("cpu"))
        before = {path: (path.read_bytes(), path.stat().st_mtime_ns)
                  for root in (results, checkpoints) for path in root.rglob("*") if path.is_file()}
        args = ["--evaluate-only", "--device", "cpu", "--experiment-name", "trial", "--models", "g05_full_reconstruction",
                "--results-root", str(results.parent), "--checkpoint-root", str(checkpoints.parent)]
        with mock.patch.object(experiment, "run_smoke_tests", side_effect=AssertionError("unexpected smoke test")), mock.patch.object(
            experiment, "train_and_evaluate_run", side_effect=AssertionError("unexpected training"),
        ), mock.patch.object(torch.optim, "AdamW", side_effect=AssertionError("unexpected optimizer")), contextlib.redirect_stdout(io.StringIO()) as output:
            experiment.main(args)
            # Explicit directories must override the name independently, without appending it.
            experiment.main([*args, "--experiment-name", "irrelevant", "--evaluation-results-dir", str(results),
                             "--evaluation-checkpoint-dir", str(checkpoints)])
        self.assertIn("Using seed-specific results directory", output.getvalue())
        self.assertIn("documentation/formatting changes accepted", output.getvalue())
        self.assertFalse((results.parent / "trial").exists())
        evaluation_paths = list((results / "evaluations").glob("*/runs/*/result.json"))
        self.assertEqual(len(evaluation_paths), 2)
        for path in evaluation_paths:
            evaluated = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(evaluated["configuration"], original["configuration"])
            context = evaluated["evaluation_context"]
            self.assertEqual(context["source_protocol_path"], str((results / "protocol.json").resolve()))
            self.assertEqual(context["checkpoint_root"], str(checkpoints.resolve()))
            self.assertEqual(context["source_compatibility"]["verification"], "identical_executable_ast")
            for selection in experiment.CHECKPOINT_SELECTIONS:
                self.assertEqual(evaluated["evaluations"][selection]["test_metrics"], original["evaluations"][selection]["test_metrics"])
        for path, (contents, modified) in before.items():
            self.assertEqual(path.read_bytes(), contents)
            self.assertEqual(path.stat().st_mtime_ns, modified)

    def test_evaluation_source_allows_only_documentation_and_format_changes(self) -> None:
        self.case_root.mkdir(parents=True)
        original = self.case_root / "original.py"
        documented = self.case_root / "documented.py"
        changed = self.case_root / "changed.py"
        original.write_text('"""Original docs."""\nthreshold = 1.0\nclass Model:\n    """Model docs."""\n    def predict(self):\n        """Predict docs."""\n        return threshold\n', encoding="utf-8")
        documented.write_text('# New comment\n"""새 설명."""\nthreshold=1.0\n\nclass Model:\n    """새 모델 설명."""\n    def predict(self):\n        """새 예측 설명."""\n        return threshold\n', encoding="utf-8")
        changed.write_text(documented.read_text(encoding="utf-8").replace("threshold=1.0", "threshold=2.0"), encoding="utf-8")
        protocol = {"source_sha256": {"NewLearning9.py": experiment.file_sha256(original)},
                    "source_ast_sha256": {"NewLearning9.py": experiment.source_ast_sha256(original)}}
        with mock.patch.object(physics, "__file__", str(documented)):
            self.assertEqual(experiment.validate_evaluation_source(protocol)["verification"], "identical_executable_ast")
            unknown_legacy = {"source_sha256": {"NewLearning9.py": "unverified-source"}}
            with self.assertRaisesRegex(RuntimeError, "compatibility cannot be verified"):
                experiment.validate_evaluation_source(unknown_legacy)
        with mock.patch.object(physics, "__file__", str(changed)):
            with self.assertRaisesRegex(RuntimeError, "executable code differs"):
                experiment.validate_evaluation_source(protocol)

    def test_evaluation_path_resolution_does_not_override_explicit_or_multi_seed_requests(self) -> None:
        protocol = self.protocol(epochs=1)
        results, checkpoints, args = self.evaluation_trial(protocol)
        candidate = results.with_name("trial_seed42")
        experiment.initialize_experiment_artifacts(experiment_results_dir=candidate, experiment_checkpoint_dir=checkpoints,
                                                  protocol=protocol, split=self.split)
        self.assertEqual(experiment.resolve_evaluation_roots(experiment.parse_args(args)), (results, checkpoints))
        (results / "protocol.json").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "evaluation-results-dir"):
            experiment.resolve_evaluation_roots(experiment.parse_args([*args, "--seeds", "42,43"]))
        with self.assertRaisesRegex(FileNotFoundError, "evaluation-results-dir"):
            experiment.resolve_evaluation_roots(experiment.parse_args([*args, "--evaluation-results-dir", str(results)]))

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


if __name__ == "__main__":
    unittest.main()
