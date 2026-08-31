"""Synthetic-fixture tests only: no optimizer, training or production writes."""
from __future__ import annotations

import contextlib
import copy
import io
import json
import os
import shutil
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest import mock

import numpy as np
import torch

import generate_charge_dataset as generator
import ModelExperiment10 as routing
import ModelExperiment11 as experiment
import NewLearning9 as physics
import visualize_3d_v2 as visualization


class VisualizationV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="viz2-tests-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, cls.old_threads)
        cls.env_patch = mock.patch.dict(os.environ, {"MPLCONFIGDIR": str(cls.root / "mpl")})
        cls.env_patch.start()
        cls.addClassCleanup(cls.env_patch.stop)
        import matplotlib
        matplotlib.use("Agg", force=True)
        cls.cpu = torch.device("cpu")
        cls.data_path = cls.root / "five.npz"
        generator.save_dataset(generator.generate_dataset(sample_count=40, g05_point_count=32,
                                                           seed=43, charge_count=5), cls.data_path)
        cls.arrays = physics.load_dataset(cls.data_path)
        cls.split = physics.create_data_split(len(cls.arrays.target))
        cls.stats = physics.calculate_normalization_stats(cls.arrays, cls.split.train)
        cls.candidate = {"id": "fixture", "learning_rate": .001, "weight_decay": .0001, "structure_dropout": .2}
        cls.source_hashes = {name: routing.file_sha256(visualization.ROOT / "Codes" / name) for name in experiment.SOURCES}
        base = routing.build_protocol(data_path=cls.data_path, arrays=cls.arrays, split=cls.split, stats=cls.stats,
                                      settings=physics.TrainingSettings(max_epochs=2, batch_size=4), device=cls.cpu,
                                      regularization=routing.RegularizationSettings(structure_dropout=.2, early_stopping_patience=0))
        cls.study = experiment.seal({
            "schema": experiment.SCHEMA, "legacy_protocol": base, "candidates": [cls.candidate],
            "data": {"path": str(cls.data_path), "sha256": routing.file_sha256(cls.data_path),
                     "array_hashes": experiment.dataset_hashes(cls.arrays)},
            "split": {name: getattr(cls.split, name).tolist() for name in ("train", "validation", "test")},
            "split_seed": physics.DATA_SPLIT_SEED, "normalization": cls.stats.to_dict(),
            "source_sha256": cls.source_hashes, "environment": experiment.environment(cls.cpu),
        })
        cls.main_study = cls.root / "main"
        cls.sweep_study = cls.root / "seed43"
        cls.configs = [experiment.configuration(cls.study, cls.candidate, model, fraction, 43)
                       for fraction in (0., .1, .25, .5, .75, 1.) for model in routing.DEFAULT_MODELS]
        for directory in (cls.main_study, cls.sweep_study):
            directory.mkdir()
            routing.atomic_write_json(directory / "normalization.json", cls.stats.to_dict())
            routing.atomic_save_npz(directory / "split_indices.npz", **{
                name: getattr(cls.split, name) for name in ("train", "validation", "test")})
        routing.atomic_write_json(cls.main_study / "study.json", cls.study)
        identity = {
            "seed": 43, "parent_study_fingerprint": cls.study["fingerprint"],
            "models": list(routing.DEFAULT_MODELS), "fractions": [0., .1, .25, .5, .75, 1.],
            "data": {**cls.study["data"], "split_seed": physics.DATA_SPLIT_SEED},
            "source_audit": {name: {"current_sha256": digest} for name, digest in cls.source_hashes.items()},
            "environment": cls.study["environment"],
            "run_configuration_fingerprints": [experiment.digest(config) for config in cls.configs],
        }
        cls.sweep = experiment.seal({"schema": visualization.SWEEP_SCHEMA, "identity": identity})
        routing.atomic_write_json(cls.sweep_study / "protocol.json", cls.sweep)
        entries, hashes = [], {}
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(43)
            for config in cls.configs:
                trial = experiment.trial_dir(cls.main_study, config)
                trial.mkdir(parents=True)
                routing.atomic_write_json(trial / "config.json", config)
                state = physics.copy_state(routing.model_from_config(config))
                observed = config["observation"]["g05_count_per_sample"] > 0
                losses = [physics.EpochLoss(total=1.5 if observed else 1., structure=1.,
                                           position=.4, magnitude=.4, relative_sign=.2, global_sign=.5 if observed else None),
                          physics.EpochLoss(total=1.4 if observed else 1.1, structure=1.1,
                                           position=.5, magnitude=.4, relative_sign=.2, global_sign=.3 if observed else None)]
                history = [{"epoch": epoch, "train": asdict(value), "validation": asdict(value)}
                           for epoch, value in enumerate(losses, 1)]
                routing.atomic_write_json(trial / "history.json", history)
                best = {}
                for epoch, value in enumerate(losses, 1):
                    routing.update_best_checkpoints(best, config=config, epoch=epoch, validation=value, model_state=state)
                evaluations = {}
                for selection, checkpoint in best.items():
                    path = trial / f"best_{selection}.pt"
                    routing.atomic_torch_save(checkpoint, path)
                    evaluations[selection] = {
                        **{key: checkpoint[key] for key in ("selected_epoch", "selected_validation_loss", "validation_losses")},
                        "checkpoint_path": path.relative_to(cls.main_study).as_posix(),
                        "checkpoint_sha256": routing.file_sha256(path), "validation_metrics": {},
                    }
                tracker = routing.replay_early_stopping(config, history)
                record = experiment.seal({
                    "schema": experiment.SCHEMA, "status": "validation_complete", "test_evaluated": False,
                    "configuration": config, "run_fingerprint": experiment.digest(config),
                    "history_sha256": routing.file_sha256(trial / "history.json"),
                    "training_result": routing.completion_metadata(config, 2, tracker.state_dict()),
                    "evaluations": evaluations,
                })
                routing.atomic_write_json(trial / "result.json", record)
                for path in trial.iterdir():
                    hashes[path.relative_to(cls.main_study).as_posix()] = routing.file_sha256(path)
                shutil.copytree(trial, experiment.trial_dir(cls.sweep_study, config))
                entries.append({"configuration": config, "trial": trial.name})
        routing.atomic_write_json(cls.main_study / "selection.json", experiment.seal({
            "schema": experiment.SCHEMA, "study_fingerprint": cls.study["fingerprint"],
            "test_used_for_selection": False, "evaluation_runs": entries, "artifact_sha256": hashes,
        }))
        routing.atomic_write_json(cls.sweep_study / "validation_selection.json", experiment.seal({
            "schema": visualization.SWEEP_SCHEMA, "protocol_identity_fingerprint": experiment.digest(identity), "seed": 43,
            "test_used_for_selection": False, "runs": entries, "artifact_sha256": hashes,
        }))

    def checkpoint(self, fraction=1., model="g05_full_reconstruction", selection="structure", *, sweep=True):
        directory = self.sweep_study if sweep else self.main_study
        trial = experiment.trial_key("fixture", model, fraction, 43)
        return directory / "runs" / trial / f"best_{selection}.pt"

    def load(self, **kwargs):
        return visualization.restore_runtime_context(self.checkpoint(**kwargs))

    def test_both_layouts_models_all_fractions_and_checkpoint_selections(self):
        for sweep in (True, False):
            for fraction in (0., .1, .25, .5, .75, 1.):
                for model in routing.DEFAULT_MODELS:
                    for selection in routing.CHECKPOINT_SELECTIONS:
                        with self.subTest(sweep=sweep, fraction=fraction, model=model, selection=selection):
                            context = self.load(fraction=fraction, model=model, selection=selection, sweep=sweep)
                            self.assertFalse(context.model.training)
                            self.assertEqual(context.model.structure_dropout.p, .2)
                            self.assertEqual(context.model.allow_g05_for_structure, model == "g05_full_reconstruction")
                            self.assertEqual(context.seed, 43)
                            self.assertEqual(context.selection, selection)
                            np.testing.assert_array_equal(context.split.test, self.split.test)
                            self.assertEqual(context.stats.to_dict(), self.stats.to_dict())
                            self.assertEqual(context.checkpoint["selected_epoch"], 2 if selection == "total" and fraction > 0 else 1)

    def test_displayed_metrics_equal_original_evaluator_and_weights_stay_frozen(self):
        for fraction in (0., .1, .25, .5, .75, 1.):
            for model in routing.DEFAULT_MODELS:
                with self.subTest(fraction=fraction, model=model):
                    context = self.load(fraction=fraction, model=model)
                    before = physics.copy_state(context.model)
                    batch = visualization.infer_test(context, batch_size=3)
                    data = physics.prepare_dataset(context.arrays, context.split.test, context.stats, fraction)
                    metrics = physics.evaluate_model(context.model, data, context.stats, batch_size=3,
                                                     weights=physics.LossWeights(**context.config["training"]["loss_weights"]))
                    for name, values in batch.per_sample_metrics.items():
                        self.assertAlmostEqual(float(values.mean(dtype=np.float64)), metrics[name], places=6)
                    self.assertEqual(batch.positions.shape, (4, 5, 3))
                    np.testing.assert_array_equal(np.sort(batch.assignment, axis=1), np.tile(np.arange(5), (4, 1)))
                    raw = context.arrays.target[batch.dataset_indices][np.arange(4)[:, None], batch.assignment]
                    np.testing.assert_allclose(batch.target_positions, raw[:, :, :3], rtol=2e-6, atol=2e-6)
                    np.testing.assert_allclose(batch.target_charges, raw[:, :, 3], rtol=2e-6, atol=2e-6)
                    for name, value in context.model.state_dict().items():
                        self.assertTrue(torch.equal(value, before[name]))
                    self.assertTrue(all(parameter.grad is None for parameter in context.model.parameters()))

    def test_matching_uses_saved_joint_weights_not_position_only(self):
        context = self.load()
        data = physics.prepare_dataset(context.arrays, context.split.test, context.stats, context.fraction)
        position, charge = data.tensors[3:]
        permutation = torch.tensor([4, 3, 2, 1, 0])
        output = physics.ModelOutput(position=position, magnitude=charge.abs()[:, permutation],
                                     relative_sign_logit=torch.zeros_like(charge), global_sign_logit=torch.zeros(len(charge)))

        class FixedModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.placeholder = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

            def forward(self, g00, g05, mask):
                return output

        checkpoint = copy.deepcopy(context.checkpoint)
        checkpoint["configuration"]["training"]["loss_weights"] = {
            "position": 0., "magnitude": 1., "relative_sign": 0., "global_sign": 1.,
        }
        # A test-only in-memory context; no production configuration is modified.
        context = replace(context, checkpoint=checkpoint, model=FixedModel())
        batch = visualization.infer_test(context, batch_size=4)
        np.testing.assert_array_equal(batch.assignment, np.tile(permutation.numpy(), (4, 1)))
        self.assertFalse(np.array_equal(batch.assignment, np.tile(np.arange(5), (4, 1))))
        np.testing.assert_allclose(batch.target_charges, context.arrays.target[context.split.test, :, 3][:, permutation],
                                   rtol=2e-6, atol=2e-6)

    def test_fraction_zero_is_a_whole_vector_equivalence_class_never_oracle_plot(self):
        context = self.load(fraction=0.)
        batch = visualization.infer_test(context)
        raw = batch.raw_charges.copy()
        for index in range(len(batch.errors)):
            sample = visualization.sample_record(context, batch, "index", index)
            self.assertFalse(sample["absolute_sign_identifiable"])
            self.assertFalse(sample["oracle_sign_used_for_plot"])
            self.assertIsNone(sample["global_sign_accuracy"])
            self.assertIsNone(sample["absolute_sign_accuracy"])
            self.assertEqual(sample["prediction_sign_representation"], "whole-vector +/- equivalence class")
            self.assertEqual(sample["metrics"]["charge_mae"], sample["metrics"]["global_invariant_charge_mae"])
        np.testing.assert_array_equal(raw, batch.raw_charges)

    def test_rejects_latest_and_modified_selected_weights(self):
        original = routing.load_torch_checkpoint(self.checkpoint(), self.cpu)
        for case in ("latest", "modified"):
            checkpoint = copy.deepcopy(original)
            if case == "latest":
                checkpoint["checkpoint_selection"] = "latest"
                checkpoint["checkpoint_kind"] = "latest"
            else:
                next(iter(checkpoint["model_state_dict"].values())).add_(.1)
            path = self.root / f"{case}.pt"
            routing.atomic_torch_save(checkpoint, path)
            with self.subTest(case=case), self.assertRaises(RuntimeError):
                visualization.restore_runtime_context(path, study_dir=self.sweep_study)

    def test_missing_selection_lock_blocks_test_data_access(self):
        original = Path.is_file
        for sweep, filename in ((True, "validation_selection.json"), (False, "selection.json")):
            with mock.patch.object(Path, "is_file", lambda path: False if path.name == filename else original(path)), \
                    mock.patch.object(physics, "load_dataset", side_effect=AssertionError("test opened before lock")), \
                    self.subTest(sweep=sweep), self.assertRaisesRegex(RuntimeError, "Test access denied"):
                self.load(sweep=sweep)

    def test_no_normalization_refit_training_or_other_checkpoint_reads(self):
        expected = self.checkpoint()
        original = routing.load_torch_checkpoint
        calls = []

        def tracked(path, device):
            calls.append(path.resolve())
            self.assertEqual(path.resolve(), expected.resolve())
            return original(path, device)

        with mock.patch.object(routing, "load_torch_checkpoint", side_effect=tracked), \
                mock.patch.object(physics, "calculate_normalization_stats", side_effect=AssertionError("refit")), \
                mock.patch.object(experiment, "train_validation_run", side_effect=AssertionError("train")), \
                mock.patch.object(routing, "run_epoch", side_effect=AssertionError("train")):
            visualization.infer_test(self.load())
        self.assertEqual(calls, [expected.resolve()])

    def test_rejects_changed_normalization_and_source(self):
        original = visualization.read_json

        def changed(path):
            value = original(path)
            if path.name == "normalization.json":
                value["charge_scale"] *= 2
            return value

        with mock.patch.object(visualization, "read_json", side_effect=changed), \
                self.assertRaisesRegex(RuntimeError, "normalization changed"):
            self.load()
        hashes = dict(self.source_hashes, **{"NewLearning9.py": "0" * 64})
        with self.assertRaisesRegex(RuntimeError, "Source snapshot/hash mismatch"):
            visualization.source_verification(self.sweep_study, hashes)

    def test_rejects_swapped_validation_test_split(self):
        path = self.sweep_study / "split_indices.npz"
        before = path.read_bytes()
        try:
            routing.atomic_save_npz(path, train=self.split.train, validation=self.split.test, test=self.split.validation)
            with self.assertRaisesRegex(RuntimeError, "Saved split differs"):
                self.load()
        finally:
            path.write_bytes(before)

    def test_accepts_identical_relocated_data_and_checkpoint_rejects_changed_data(self):
        path = self.root / "relocated five.npz"
        moved = self.root / "moved checkpoint.pt"
        shutil.copyfile(self.data_path, path)
        shutil.copyfile(self.checkpoint(), moved)
        context = visualization.restore_runtime_context(moved, path, self.cpu, self.sweep_study)
        self.assertEqual(context.provenance["data_path"], str(path))
        with path.open("ab") as handle:
            handle.write(b"different bytes")
        with self.assertRaisesRegex(RuntimeError, "SHA256 differs"):
            visualization.restore_runtime_context(moved, path, self.cpu, self.sweep_study)

    def test_main_uses_saved_index_list_when_optional_split_archive_is_absent(self):
        original = Path.is_file
        with mock.patch.object(Path, "is_file", lambda path: False if path == self.main_study / "split_indices.npz" else original(path)), \
                mock.patch.object(physics, "create_data_split", side_effect=AssertionError("regenerated main split")):
            context = self.load(sweep=False)
        np.testing.assert_array_equal(context.split.test, self.split.test)

    def test_unique_outputs_have_reproducible_metadata_and_do_not_modify_inputs(self):
        context = self.load(fraction=0.)
        batch = visualization.infer_test(context)
        before = {path: routing.file_sha256(path) for path in self.sweep_study.rglob("*") if path.is_file()}
        outputs = []
        for mode in ("all", "median"):
            with contextlib.redirect_stdout(io.StringIO()):
                outputs.append(visualization.export_visualizations(context, batch, output_dir=self.root / "figures",
                                mode=mode, show_g00=True, show_g05=True, dpi=72))
        self.assertNotEqual(outputs[0], outputs[1])
        self.assertEqual(len(list(outputs[0].glob("*.png"))), 3)
        for directory in outputs:
            self.assertIn("seed43", directory.parts)
            manifest = visualization.read_json(directory / "manifest.json")
            self.assertEqual(manifest["configuration"], context.config)
            self.assertEqual(manifest["provenance"]["checkpoint_sha256"], routing.file_sha256(context.checkpoint_path))
            for figure in manifest["figures"]:
                self.assertEqual(routing.file_sha256(directory / figure["png"]), figure["sha256"])
                self.assertFalse(visualization.read_json(directory / figure["sample_metadata"])["sample"]["oracle_sign_used_for_plot"])
            with np.load(directory / manifest["predictions"], allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["raw_charges"], batch.raw_charges)
                np.testing.assert_array_equal(saved["assignment"], batch.assignment)
        self.assertEqual(before, {path: routing.file_sha256(path) for path in before})
        with self.assertRaises(FileExistsError):
            visualization.write_json_exclusive(outputs[0] / "manifest.json", {})

    def test_sample_selection_is_stable_and_index_is_within_saved_test_split(self):
        batch = visualization.infer_test(self.load())
        metrics = dict(batch.per_sample_metrics, mean_position_3d_error=np.array([4., 1., 2., 2.]))
        batch = replace(batch, per_sample_metrics=metrics)
        self.assertEqual(visualization.select_samples(batch, "all"), [("best", 1), ("median", 2), ("worst", 0)])
        self.assertEqual(visualization.select_samples(batch, "all", 3), [("index", 3)])
        with self.assertRaises(ValueError):
            visualization.select_samples(batch, "all", 4)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            visualization.parse_args(["--checkpoint", "x", "--sample-index", "0", "--sample-mode", "all"])

    def test_check_only_creates_no_visualization_and_does_no_inference(self):
        output = self.root / "check-only-output"
        with mock.patch.object(visualization, "infer_test", side_effect=AssertionError("check-only inferred")), \
                contextlib.redirect_stdout(io.StringIO()):
            result = visualization.main(["--checkpoint", str(self.checkpoint()), "--device", "cpu", "--check-only", "--output-dir", str(output)])
        self.assertEqual(result, 0)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
