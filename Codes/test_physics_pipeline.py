from __future__ import annotations

import itertools
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

import generate_charge_dataset as generator

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    import NewLearning9 as five_charge

try:
    import train_g05_fraction_experiment as experiment
except ModuleNotFoundError:
    experiment = None


class DatasetGenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = generator.generate_dataset(
            sample_count=12,
            g05_point_count=32,
            seed=123,
        )

    def test_shapes_and_documented_target_order(self) -> None:
        self.assertEqual(self.dataset["G00"].shape, (12, 32, 32))
        self.assertEqual(self.dataset["G05"].shape, (12, 32, 3))
        self.assertEqual(self.dataset["target"].shape, (12, 8))
        self.assertEqual(
            tuple(str(value) for value in self.dataset["target_fields"]),
            generator.TARGET_FIELDS,
        )
        self.assertTrue(
            np.all(self.dataset["target"][:, 0] <= self.dataset["target"][:, 4])
        )

    def test_candidate_sensors_are_fixed_and_unique(self) -> None:
        sensor_positions = self.dataset["G05"][:, :, 0:2]
        self.assertTrue(np.all(sensor_positions == sensor_positions[0:1]))
        self.assertEqual(np.unique(sensor_positions[0], axis=0).shape[0], 32)

    def test_five_charge_generation_preserves_tuple_fields(self) -> None:
        dataset = generator.generate_dataset(sample_count=12, g05_point_count=16, seed=123, charge_count=5)
        self.assertEqual(dataset["target"].shape, (12, 20))
        self.assertEqual(tuple(dataset["target_fields"][-4:]), ("x5", "y5", "z5", "q5"))
        self.assertEqual(int(dataset["charge_count"]), 5)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            generator.generate_dataset(charge_count=0)

    def test_g00_and_g05_follow_the_same_signed_potential(self) -> None:
        target = self.dataset["target"][0]
        grid_axis = self.dataset["grid_x"]
        grid_x, grid_y = np.meshgrid(grid_axis, grid_axis)
        charge_x = target[[0, 4]]
        charge_y = target[[1, 5]]
        charge_z = target[[2, 6]]
        charge_q = target[[3, 7]]
        potential = generator.coulomb_potential(
            grid_x,
            grid_y,
            charge_x,
            charge_y,
            charge_z,
            charge_q,
        )
        np.testing.assert_allclose(
            potential**2,
            self.dataset["G00"][0],
            rtol=2e-5,
            atol=2e-7,
        )
        x_index = self.dataset["G05"][0, :, 0].astype(np.int64)
        y_index = self.dataset["G05"][0, :, 1].astype(np.int64)
        np.testing.assert_allclose(
            potential[y_index, x_index],
            self.dataset["G05"][0, :, 2],
            rtol=2e-5,
            atol=2e-7,
        )


@unittest.skipIf(experiment is None, "Legacy train_g05_fraction_experiment module is unavailable")
class TrainingSemanticsTests(unittest.TestCase):
    def test_default_fraction_counts_are_nested(self) -> None:
        counts = [
            experiment.g05_count_for_fraction(fraction, 32)
            for fraction in experiment.G05_FRACTIONS
        ]
        self.assertEqual(counts, [0, 3, 8, 16, 24, 32])
        masks = [
            experiment.create_g05_mask(2, 32, fraction)
            for fraction in experiment.G05_FRACTIONS
        ]
        for lower, higher in zip(masks, masks[1:]):
            self.assertTrue(np.all(lower <= higher))

    def test_charge_loss_is_selected_per_sample(self) -> None:
        prediction = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
        target = torch.tensor([[-1.0, -1.0], [-1.0, -1.0]])
        mask = torch.zeros((2, 1, 1))
        mask[0, 0, 0] = 1.0
        loss = experiment.samplewise_charge_mse(prediction, target, mask)
        # Observed sample: signed MSE=4. Missing sample: invariant MSE=0.
        self.assertAlmostEqual(loss.item(), 2.0)

    def test_model_capacity_does_not_depend_on_fraction(self) -> None:
        experiment.set_reproducibility(41)
        first = experiment.ChargeNet()
        experiment.set_reproducibility(41)
        second = experiment.ChargeNet()
        for first_parameter, second_parameter in zip(
            first.parameters(),
            second.parameters(),
        ):
            self.assertTrue(torch.equal(first_parameter, second_parameter))


@unittest.skipIf(torch is None, "PyTorch is not installed in this Python environment")
class FiveChargeTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.generated = generator.generate_dataset(sample_count=12, g05_point_count=16, seed=321, charge_count=5)
        cls.arrays = five_charge.DatasetArrays(
            g00=cls.generated["G00"], g05=cls.generated["G05"],
            target=cls.generated["target"].reshape(12, 5, 4),
            grid_x=cls.generated["grid_x"], grid_y=cls.generated["grid_y"],
        )
        cls.split = five_charge.create_data_split(12)
        cls.stats = five_charge.calculate_normalization_stats(cls.arrays, cls.split.train)
        cls.dataset = five_charge.prepare_dataset(cls.arrays, np.arange(12), cls.stats, 1.0)

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls.previous_threads)

    def make_prediction(self, count: int = 2) -> five_charge.ModelOutput:
        torch.manual_seed(19)
        positions, charges = self.dataset.tensors[3:]
        return five_charge.ModelOutput(
            position=(positions[:count] + torch.randn(count, 5, 3) * 0.3).requires_grad_(),
            magnitude=(charges[:count].abs() + torch.rand(count, 5) * 0.2).requires_grad_(),
            relative_sign_logit=torch.randn(count, 5, requires_grad=True),
            global_sign_logit=torch.randn(count, requires_grad=True),
        )

    def test_loader_accepts_unsorted_flat_and_grouped_targets(self) -> None:
        rng = np.random.default_rng(89)
        permutations = np.stack([rng.permutation(5) for _ in range(12)])
        shuffled = np.take_along_axis(self.arrays.target, permutations[:, :, None], axis=1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "five.npz"
            for target in (shuffled, shuffled.reshape(12, 20)):
                generator.save_dataset({**self.generated, "target": target}, path)
                loaded = five_charge.load_dataset(path)
                np.testing.assert_array_equal(loaded.target, shuffled)
            invalid = generator.generate_dataset(sample_count=12, g05_point_count=16, charge_count=2)
            generator.save_dataset(invalid, path)
            with self.assertRaisesRegex(ValueError, "five charges"):
                five_charge.load_dataset(path)

    def test_normalization_is_shared_order_invariant_and_train_only(self) -> None:
        shuffled = replace(self.arrays, target=self.arrays.target[:, [4, 2, 0, 3, 1]])
        stats = five_charge.calculate_normalization_stats(shuffled, self.split.train)
        self.assertEqual(stats.position_mean.shape, (3,))
        np.testing.assert_array_equal(stats.position_mean, self.stats.position_mean)
        np.testing.assert_array_equal(stats.position_std, self.stats.position_std)
        self.assertEqual(stats.charge_scale, self.stats.charge_scale)
        held_out = np.concatenate((self.split.validation, self.split.test))
        changed = replace(self.arrays, target=self.arrays.target.copy(), g00=self.arrays.g00.copy(), g05=self.arrays.g05.copy())
        for array in (changed.target, changed.g00, changed.g05):
            array[held_out] += 1000
        changed_stats = five_charge.calculate_normalization_stats(changed, self.split.train)
        self.assertEqual(changed_stats.to_dict(), self.stats.to_dict())

    def test_assignment_is_exact_and_uses_each_target_once(self) -> None:
        rng = np.random.default_rng(71)
        costs = rng.normal(size=(4, 5, 5)).astype(np.float32)
        # All rows prefer target zero. Independent nearest neighbors would fail.
        costs[0, :, 0] = -10
        assignment = five_charge.minimum_cost_assignment(torch.from_numpy(costs)).numpy()
        for cost, chosen in zip(costs, assignment):
            self.assertEqual(sorted(chosen.tolist()), list(range(5)))
            expected = min(sum(float(cost[row, column]) for row, column in enumerate(p))
                           for p in itertools.permutations(range(5)))
            actual = sum(float(cost[row, column]) for row, column in enumerate(chosen))
            self.assertAlmostEqual(actual, expected, places=6)

    def test_all_120_target_orders_have_identical_losses(self) -> None:
        output = self.make_prediction()
        mask, positions, charges = (t[:2] for t in self.dataset.tensors[2:])
        reference = five_charge.calculate_losses(output, positions, charges, mask)
        for permutation in itertools.permutations(range(5)):
            order = list(permutation)
            losses = five_charge.calculate_losses(output, positions[:, order], charges[:, order], mask)
            for name in ("total", "structure", "position", "magnitude", "relative_sign", "global_sign"):
                torch.testing.assert_close(getattr(losses, name), getattr(reference, name), rtol=1e-6, atol=1e-6)

    def test_matching_minimizes_the_actual_joint_training_objective(self) -> None:
        output = self.make_prediction()
        mask, positions, charges = (t[:2] for t in self.dataset.tensors[2:])
        weights = five_charge.LossWeights(position=1.7, magnitude=0.8, relative_sign=0.35)
        permutations = torch.tensor(list(itertools.permutations(range(5))))
        relative, _ = five_charge.canonical_sign_targets(charges)
        candidate_position = (output.position[:, None] - positions[:, permutations]).square().mean(dim=(-2, -1))
        candidate_magnitude = (output.magnitude[:, None] - charges.abs()[:, permutations]).square().mean(dim=-1)
        candidate_relative = five_charge.relative_sign_nll(
            output.relative_sign_logit[:, None].expand(-1, 120, -1).reshape(-1, 5),
            relative[:, permutations].reshape(-1, 5),
        ).reshape(2, 120)
        candidate_objectives = (weights.position * candidate_position + weights.magnitude * candidate_magnitude
                                + weights.relative_sign * candidate_relative)
        expected = candidate_objectives.min(dim=1).values.mean()
        actual = five_charge.calculate_losses(output, positions, charges, mask, weights)
        torch.testing.assert_close(actual.structure, expected)

    def test_prediction_permutation_preserves_loss_and_permutes_gradients(self) -> None:
        output = self.make_prediction()
        mask, positions, charges = (t[:2] for t in self.dataset.tensors[2:])
        loss = five_charge.calculate_losses(output, positions, charges, mask)
        names = ("position", "magnitude", "relative_sign_logit", "global_sign_logit")
        gradients = torch.autograd.grad(loss.total, tuple(getattr(output, name) for name in names))
        order = [3, 0, 4, 1, 2]
        permuted = five_charge.ModelOutput(
            **{name: (getattr(output, name)[:, order] if name != "global_sign_logit" else getattr(output, name))
               .detach().clone().requires_grad_() for name in names}
        )
        permuted_loss = five_charge.calculate_losses(permuted, positions, charges, mask)
        torch.testing.assert_close(loss.total, permuted_loss.total)
        new_gradients = torch.autograd.grad(permuted_loss.total, tuple(getattr(permuted, name) for name in names))
        for name, first, second in zip(names, gradients, new_gradients):
            torch.testing.assert_close(second, first[:, order] if name != "global_sign_logit" else first)

    def test_global_sign_anchor_and_missing_g05_symmetry(self) -> None:
        charges = torch.tensor([[-2.0, 3.0, -4.0, 5.0, -6.0]])
        relative, global_sign = five_charge.canonical_sign_targets(charges)
        torch.testing.assert_close(relative, torch.tensor([[1.0, 0.0, 1.0, 0.0, 1.0]]))
        torch.testing.assert_close(global_sign, torch.zeros(1))
        flipped_relative, flipped_global = five_charge.canonical_sign_targets(-charges)
        torch.testing.assert_close(relative, flipped_relative)
        torch.testing.assert_close(flipped_global, 1 - global_sign)
        position = torch.arange(15, dtype=torch.float32).reshape(1, 5, 3)
        output = five_charge.ModelOutput(position, charges.abs(), (relative * 2 - 1) * 20, torch.ones(1))
        mask = torch.zeros(1, 16, 1)
        exact = five_charge.calculate_losses(output, position, charges, mask)
        flipped = five_charge.calculate_losses(output, position, -charges, mask)
        self.assertIsNone(exact.global_sign)
        torch.testing.assert_close(exact.total, flipped.total)
        wrong = charges.clone()
        wrong[:, 0] *= -1
        incorrect = five_charge.calculate_losses(output, position, wrong, mask)
        self.assertGreater(incorrect.structure.item(), exact.structure.item() + 0.1)

    def test_relative_decoder_enforces_parity_and_maximum_likelihood(self) -> None:
        raw = torch.tensor(list(itertools.product((-1.0, 1.0), repeat=5)))
        logits = raw * torch.tensor([0.2, 0.7, 1.5, 2.1, 3.3])
        decoded = five_charge.decode_relative_signs(logits)
        torch.testing.assert_close(decoded.prod(dim=1), torch.ones(32))
        valid = raw[raw.prod(dim=1) == 1]
        costs = torch.nn.functional.binary_cross_entropy_with_logits(
            logits[:, None, :].expand(-1, 16, -1), ((valid + 1) / 2)[None].expand(32, -1, -1),
            reduction="none",
        ).sum(dim=-1)
        torch.testing.assert_close(decoded, valid[costs.argmin(dim=1)])
        nll = five_charge.relative_sign_nll(logits, (decoded + 1) / 2)
        expected = (-costs).logsumexp(dim=1) + costs.min(dim=1).values
        torch.testing.assert_close(nll, expected / 5, atol=1e-6, rtol=1e-5)

    def test_sign_only_forward_masking_and_gradient_isolation(self) -> None:
        five_charge.set_reproducibility(17)
        model = five_charge.ChargeNet()
        g00, g05, mask, positions, charges = (t[:2].clone() for t in self.dataset.tensors)
        mask[:, 5:] = 0
        output = model(g00, g05, mask)
        reversed_output = model(g00, g05 * torch.tensor([1, 1, -1]), mask)
        changed_g00 = model(g00 + 100, g05, mask)
        for name in ("position", "magnitude", "relative_sign_logit"):
            torch.testing.assert_close(getattr(output, name), getattr(reversed_output, name), atol=0, rtol=0)
        torch.testing.assert_close(output.global_sign_logit, -reversed_output.global_sign_logit, atol=0, rtol=0)
        torch.testing.assert_close(output.global_sign_logit, changed_g00.global_sign_logit, atol=0, rtol=0)
        g05[:, 5:] = float("nan")
        hidden = model(g00, g05, mask)
        torch.testing.assert_close(output.global_sign_logit, hidden.global_sign_logit, atol=0, rtol=0)
        zero_mask = model(g00, g05, torch.zeros_like(mask))
        torch.testing.assert_close(zero_mask.global_sign_logit, torch.zeros(2), atol=0, rtol=0)
        losses = five_charge.calculate_losses(output, positions, charges, mask)
        losses.global_sign.backward(retain_graph=True)
        self.assertTrue(all(p.grad is None for name, p in model.named_parameters()
                            if name.startswith(five_charge.STRUCTURE_PREFIXES)))
        self.assertTrue(any(p.grad is not None and torch.any(p.grad != 0) for name, p in model.named_parameters()
                            if name.startswith(five_charge.GLOBAL_SIGN_PREFIXES)))
        model.zero_grad(set_to_none=True)
        losses.structure.backward()
        self.assertTrue(all(p.grad is None for name, p in model.named_parameters()
                            if name.startswith(five_charge.GLOBAL_SIGN_PREFIXES)))

    def test_global_only_training_matches_joint_training_with_missing_samples(self) -> None:
        five_charge.set_reproducibility(29)
        reference = five_charge.ChargeNet()
        global_only = five_charge.ChargeNet()
        global_only.load_state_dict(reference.state_dict())
        initial_structure = five_charge.copy_state(global_only, five_charge.STRUCTURE_PREFIXES)
        g00, g05, mask, positions, charges = self.dataset.tensors
        mask = mask.clone()
        mask[:4] = 0  # Include an entirely unobserved batch before the first update.
        mask[5::3] = 0
        mixed = torch.utils.data.TensorDataset(g00, g05, mask, positions, charges)
        reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=1e-3)
        global_optimizer = torch.optim.AdamW(
            [parameter for name, parameter in global_only.named_parameters()
             if name.startswith(five_charge.GLOBAL_SIGN_PREFIXES)], lr=1e-3,
        )
        weights = five_charge.LossWeights(global_sign=0.7)
        expected = five_charge.run_epoch(
            reference, five_charge.create_data_loader(mixed, 3, device=torch.device("cpu")),
            reference_optimizer, weights,
        )
        actual = five_charge.run_global_sign_epoch(
            global_only, five_charge.create_data_loader(mixed, 3, device=torch.device("cpu")),
            global_optimizer, weights.global_sign,
        )
        self.assertEqual(actual, expected.global_sign)
        expected_sign = five_charge.copy_state(reference, five_charge.GLOBAL_SIGN_PREFIXES)
        for name, tensor in five_charge.copy_state(global_only).items():
            expected_tensor = initial_structure[name] if name in initial_structure else expected_sign[name]
            torch.testing.assert_close(tensor, expected_tensor, rtol=0, atol=0)
        missing = torch.utils.data.TensorDataset(g00, g05, torch.zeros_like(mask), positions, charges)
        previous = five_charge.copy_state(global_only)
        self.assertIsNone(five_charge.run_global_sign_epoch(
            global_only, five_charge.create_data_loader(missing, 3, device=torch.device("cpu")),
            global_optimizer,
        ))
        for name, tensor in previous.items():
            torch.testing.assert_close(tensor, global_only.state_dict()[name], rtol=0, atol=0)

    def test_evaluation_is_order_invariant_and_missing_signs_are_na(self) -> None:
        five_charge.set_reproducibility(23)
        model = five_charge.ChargeNet()
        g00, g05, mask, positions, charges = self.dataset.tensors
        mask = mask.clone()
        mask[::2] = 0
        mixed = torch.utils.data.TensorDataset(g00, g05, mask, positions, charges)
        order = [4, 1, 3, 0, 2]
        shuffled = torch.utils.data.TensorDataset(g00, g05, mask, positions[:, order], charges[:, order])
        reference = five_charge.evaluate_model(model, mixed, self.stats, batch_size=5)
        actual = five_charge.evaluate_model(model, shuffled, self.stats, batch_size=5)
        self.assertEqual(reference["observed_sample_fraction"], 0.5)
        self.assertEqual(reference["observations_per_sample"], 8.0)
        for name in reference:
            self.assertAlmostEqual(actual[name], reference[name], places=6)
        missing = five_charge.prepare_dataset(self.arrays, np.arange(12), self.stats, 0.0)
        result = five_charge.evaluate_model(model, missing, self.stats)
        for name in ("global_sign_accuracy", "global_sign_bce", "absolute_sign_accuracy", "absolute_sign_set_accuracy"):
            self.assertIsNone(result[name])

    def test_nested_sensor_counts_and_fraction_rejection(self) -> None:
        counts = [five_charge.g05_count_for_fraction(f, 32) for f in five_charge.G05_FRACTIONS]
        self.assertEqual(counts, [0, 3, 8, 16, 24, 32])
        for fraction in (-0.1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                five_charge.g05_count_for_fraction(fraction, 32)

    def test_spatial_pool_matches_adaptive_averages_and_gradients(self) -> None:
        pool = five_charge.SpatialAveragePool()
        for shape in ((2, 3, 8, 8), (2, 3, 5, 7), (2, 3, 1, 2)):
            features = torch.randn(*shape, requires_grad=True)
            expected = torch.nn.functional.adaptive_avg_pool2d(features, (4, 4))
            actual = pool(features)
            torch.testing.assert_close(actual, expected)
            gradient = torch.randn_like(actual)
            expected_grad = torch.autograd.grad(expected, features, gradient)[0]
            actual_grad = torch.autograd.grad(actual, features, gradient)[0]
            torch.testing.assert_close(actual_grad, expected_grad)


if __name__ == "__main__":
    unittest.main()
