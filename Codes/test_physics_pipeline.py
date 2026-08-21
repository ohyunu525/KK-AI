from __future__ import annotations

import unittest

import numpy as np

import generate_charge_dataset as generator

try:
    import torch

    import train_g05_fraction_experiment as experiment
except ModuleNotFoundError:
    torch = None
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


@unittest.skipIf(torch is None, "PyTorch is not installed in this Python environment")
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


if __name__ == "__main__":
    unittest.main()
