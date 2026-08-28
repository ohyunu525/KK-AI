from __future__ import annotations

import contextlib
import csv
import io
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

# The experiment filename includes a dot, so it is not an importable module name.
_spec = importlib.util.spec_from_file_location(
    "model_experiment_checkpoint_tests", Path(__file__).with_name("ModelExperiment8.5.py")
)
assert _spec is not None and _spec.loader is not None
experiment = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = experiment
_spec.loader.exec_module(experiment)
import NewLearning9 as five_charge
import generate_charge_dataset as generator


class CheckpointLoadingTests(unittest.TestCase):
    def test_rng_states_remain_on_cpu_when_target_device_is_cuda(self) -> None:
        shuffle_generator = torch.Generator().manual_seed(41)
        checkpoint = {
            "shuffle_generator_state": shuffle_generator.get_state(),
            "rng_state": experiment.capture_rng_state(),
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "checkpoint.pt"
            torch.save(checkpoint, path)
            loaded = experiment.load_torch_checkpoint(path, torch.device("cuda"))

        self.assertEqual(loaded["shuffle_generator_state"].device.type, "cpu")
        self.assertEqual(loaded["rng_state"]["torch"].device.type, "cpu")

        restored_generator = torch.Generator()
        restored_generator.set_state(loaded["shuffle_generator_state"])
        experiment.restore_rng_state(loaded["rng_state"])


class FiveChargeCheckpointTests(unittest.TestCase):
    def test_best_cpu_snapshot_does_not_alias_live_parameters(self) -> None:
        model = five_charge.ChargeNet()
        snapshot = five_charge.copy_state(model)
        name, parameter = next(iter(model.named_parameters()))
        original = parameter.detach().clone()
        with torch.no_grad():
            parameter.add_(1)
        torch.testing.assert_close(snapshot[name], original, atol=0, rtol=0)

    def test_training_roundtrip_and_g05_independent_structure_updates(self) -> None:
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
                root = Path(directory)
                data_path = root / "dataset.npz"
                generator.save_dataset(generator.generate_dataset(
                    sample_count=20, g05_point_count=8, seed=91, charge_count=5,
                ), data_path)
                arrays = five_charge.load_dataset(data_path)
                split = five_charge.create_data_split(20)
                stats = five_charge.calculate_normalization_stats(arrays, split.train)
                metadata = five_charge.checkpoint_metadata(arrays, stats, split, data_path)
                trained_states = []
                training_results = []
                settings = five_charge.TrainingSettings(max_epochs=2, batch_size=8)
                weights = five_charge.LossWeights(global_sign=1.7)
                for fraction in (0.0, 1.0):
                    train = five_charge.prepare_dataset(arrays, split.train, stats, fraction)
                    validation = five_charge.prepare_dataset(arrays, split.validation, stats, fraction)
                    training = five_charge.train_model(
                        train, validation, fraction, 71, metadata, checkpoint_dir=root / "checkpoints",
                        settings=settings, weights=weights,
                        device=torch.device("cpu"),
                    )
                    training_results.append(training)
                    loaded, loaded_stats, checkpoint = five_charge.load_trained_model(training.checkpoint_path, torch.device("cpu"))
                    trained_states.append(five_charge.copy_state(training.model, five_charge.STRUCTURE_PREFIXES))
                    self.assertEqual(loaded_stats.to_dict(), stats.to_dict())
                    with torch.no_grad():
                        expected = training.model(*validation.tensors[:3])
                        actual = loaded(*validation.tensors[:3])
                    for name in ("position", "magnitude", "relative_sign_logit", "global_sign_logit"):
                        torch.testing.assert_close(getattr(actual, name), getattr(expected, name), rtol=0, atol=0)
                    component = torch.load(training.checkpoint_path.parent / "best_structure.pt", weights_only=True)
                    for name, tensor in component["component_state_dict"].items():
                        torch.testing.assert_close(tensor, checkpoint["model_state_dict"][name], rtol=0, atol=0)
                    sign_path = training.checkpoint_path.parent / "best_global_sign.pt"
                    self.assertEqual(sign_path.exists(), fraction > 0)
                    if fraction > 0:
                        sign_component = torch.load(sign_path, weights_only=True)
                        for name, tensor in sign_component["component_state_dict"].items():
                            torch.testing.assert_close(tensor, checkpoint["model_state_dict"][name], rtol=0, atol=0)
                    else:
                        self.assertIsNone(checkpoint["best_global_sign_loss"])
                # Changing the G05 fraction must not even alter the trained G00
                # weights when data, initialization and shuffle order are fixed.
                for name in trained_states[0]:
                    torch.testing.assert_close(trained_states[0][name], trained_states[1][name], rtol=0, atol=0)
                source = training_results[0].checkpoint_path.parent / "best_structure.pt"
                # Reuse must skip both the forward and backward work of the CNN.
                with mock.patch.object(torch.nn.Conv2d, "forward", side_effect=AssertionError("G00 was recomputed")):
                    reused = five_charge.train_model(
                        train, validation, 1.0, 71, metadata, checkpoint_dir=root / "reused",
                        settings=settings, weights=weights, device=torch.device("cpu"), structure_source=source,
                    )
                reference = training_results[1]
                for name, tensor in reference.model.state_dict().items():
                    torch.testing.assert_close(tensor, reused.model.state_dict()[name], rtol=0, atol=0)
                for name in ("best_structure_loss", "best_structure_epoch", "best_global_sign_loss", "best_global_sign_epoch"):
                    self.assertEqual(getattr(reference, name), getattr(reused, name))
                loaded, _, checkpoint = five_charge.load_trained_model(reused.checkpoint_path, torch.device("cpu"))
                self.assertEqual(checkpoint["structure_source"], str(source.resolve()))
                for name, tensor in reused.model.state_dict().items():
                    torch.testing.assert_close(tensor, loaded.state_dict()[name], rtol=0, atol=0)
                for seed, candidate_settings, candidate_metadata in (
                    (72, settings, metadata),
                    (71, five_charge.TrainingSettings(max_epochs=1, batch_size=8), metadata),
                    (71, settings, {**metadata, "data_path": "different-dataset.npz"}),
                ):
                    with self.subTest(seed=seed, settings=candidate_settings, data=candidate_metadata["data_path"]):
                        with self.assertRaisesRegex(ValueError, "Incompatible structure checkpoint"):
                            five_charge.train_model(
                                train, validation, 1.0, seed, candidate_metadata,
                                checkpoint_dir=root / "incompatible", settings=candidate_settings,
                                weights=weights, device=torch.device("cpu"), structure_source=source,
                            )
                # Direct API calls must not overwrite an existing seed/fraction either.
                previous_bytes = reused.checkpoint_path.read_bytes()
                with self.assertRaises(FileExistsError):
                    five_charge.train_model(
                        train, validation, 1.0, 71, metadata, checkpoint_dir=root / "reused",
                        settings=settings, weights=weights, device=torch.device("cpu"), structure_source=source,
                    )
                self.assertEqual(reused.checkpoint_path.read_bytes(), previous_bytes)
        finally:
            torch.set_num_threads(previous_threads)


    def test_partial_rerun_preserves_prior_results_and_legacy_artifacts(self) -> None:
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
                root = Path(directory)
                data_path = root / "dataset.npz"
                generator.save_dataset(generator.generate_dataset(
                    sample_count=20, g05_point_count=8, seed=91, charge_count=5,
                ), data_path)
                checkpoint_root, results_root = root / "checkpoints", root / "results"
                checkpoint_root.mkdir()
                results_root.mkdir()
                # Existing files from the old layout must remain untouched, too.
                legacy_checkpoint = checkpoint_root / "normalization.npz"
                legacy_results = results_root / "runs.csv"
                legacy_checkpoint.write_bytes(b"existing normalization")
                legacy_results.write_bytes(b"existing results")
                args = ["--data", str(data_path), "--epochs", "1", "--batch-size", "8",
                        "--device", "cpu", "--no-plots", "--checkpoint-dir", str(checkpoint_root),
                        "--results-dir", str(results_root)]
                five_charge.main(args + ["--seeds", "41,42", "--fractions", "0,1"])
                first_results = next(path for path in results_root.iterdir() if path.is_dir())
                first_checkpoints = checkpoint_root / first_results.name
                self.assertTrue(first_checkpoints.is_dir())
                before = {path: path.read_bytes() for base in (checkpoint_root, results_root)
                          for path in base.rglob("*") if path.is_file()}
                five_charge.main(args + ["--seeds", "41", "--fractions", "1", "--epochs", "2"])
                for path, content in before.items():
                    self.assertEqual(path.read_bytes(), content, str(path))
                directories = [path for path in results_root.iterdir() if path.is_dir()]
                self.assertEqual(len(directories), 2)
                second_results = next(path for path in directories if path != first_results)
                second_checkpoints = checkpoint_root / second_results.name
                self.assertTrue(second_checkpoints.is_dir())
                for result_dir, expected_pairs, expected_epochs in (
                    (first_results, {(str(s), str(f)) for s in (41, 42) for f in (0.0, 1.0)}, 1),
                    (second_results, {("41", "1.0")}, 2),
                ):
                    with (result_dir / "runs.csv").open(encoding="utf-8-sig", newline="") as handle:
                        rows = list(csv.DictReader(handle))
                    self.assertEqual({(row["seed"], row["g05_fraction"]) for row in rows}, expected_pairs)
                    for row in rows:
                        self.assertEqual(Path(row["checkpoint_path"]).parents[1], checkpoint_root / result_dir.name)
                    with (result_dir / "summary.csv").open(encoding="utf-8-sig", newline="") as handle:
                        summary = list(csv.DictReader(handle))
                    self.assertEqual({row["g05_fraction"] for row in summary}, {pair[1] for pair in expected_pairs})
                    protocol = json.loads((result_dir / "protocol.json").read_text(encoding="utf-8"))
                    self.assertEqual(protocol["run_id"], result_dir.name)
                    self.assertEqual(protocol["settings"]["max_epochs"], expected_epochs)
                self.assertEqual(legacy_checkpoint.read_bytes(), b"existing normalization")
                self.assertEqual(legacy_results.read_bytes(), b"existing results")
        finally:
            torch.set_num_threads(previous_threads)


    def test_run_directories_allow_a_shared_root_without_reusing_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_id, first_checkpoints, first_results = five_charge.create_run_directories(root, root)
            second_id, second_checkpoints, second_results = five_charge.create_run_directories(root, root)
            self.assertNotEqual(first_id, second_id)
            self.assertEqual(first_checkpoints, first_results)
            self.assertEqual(second_checkpoints, second_results)
            self.assertTrue(first_results.is_dir())
            self.assertTrue(second_results.is_dir())


if __name__ == "__main__":
    unittest.main()
