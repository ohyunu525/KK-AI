"""Behavioral tests for leakage barriers, fair comparison and interrupted resume."""
from __future__ import annotations

import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import ModelExperiment11 as tuning
import ModelExperiment10 as legacy
import NewLearning9 as physics
import generate_charge_dataset as generator
import numpy as np
import torch


class TuningPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, cls.old_threads)
        cls.temporary = tempfile.TemporaryDirectory(prefix="m11-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        cls.data_path = cls.root / "five.npz"
        generator.save_dataset(generator.generate_dataset(sample_count=40, g05_point_count=32,
                                                           seed=127, charge_count=5), cls.data_path)
        cls.default = tuning.read_json(tuning.DEFAULT_CONFIG)

    def setUp(self):
        self.directory = self.root / self._testMethodName.removeprefix("test_")[:45]
        self.spec = copy.deepcopy(self.default)
        self.spec.update(max_epochs=3, batch_size=8, early_stopping_patience=0, top_k=1,
                         screen_seeds=[41], confirmation_seeds=[42], random_candidates=0,
                         min_improvement_pct=0.0)
        self.spec["anchors"] = self.spec["anchors"][:2]
        self.spec["fresh_test"] = {"samples": 20, "seed": 991}
        self.cpu = torch.device("cpu")

    def init(self, directory=None, device=None):
        with contextlib.redirect_stdout(io.StringIO()):
            return tuning.initialize(directory or self.directory, self.spec, self.data_path, device or self.cpu)

    def config(self, study, candidate=1, model="g05_full_reconstruction"):
        return tuning.configuration(study, study["candidates"][candidate], model, 1.0, 41)

    def run_trial(self, directory, config, data, device=None):
        with contextlib.redirect_stdout(io.StringIO()):
            return tuning.train_validation_run(directory, config, data, device or self.cpu)

    def assert_tree_equal(self, a, b):
        if torch.is_tensor(a):
            self.assertTrue(torch.equal(a.cpu(), b.cpu()))
        elif isinstance(a, np.ndarray):
            self.assertTrue(np.array_equal(a, b))
        elif isinstance(a, dict):
            self.assertEqual(set(a), set(b))
            for key in a:
                self.assert_tree_equal(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b):
                self.assert_tree_equal(x, y)
        else:
            self.assertEqual(a, b)

    def test_search_reproducible_and_rejects_unfair_configuration(self):
        original_state = np.random.get_state()
        first = tuning.make_candidates(self.default)
        self.assert_tree_equal(np.random.get_state(), original_state)
        np.random.seed(997)
        self.assertEqual(first, tuning.make_candidates(self.default))
        self.assertEqual(len(first), 8)
        for field, value in (("max_epochs", 0), ("top_k", 99), ("fractions", [1.0, 1.0]),
                             ("confirmation_seeds", [41]), ("models", ["g05_full_reconstruction"]),
                             ("random_candidates", -1)):
            bad = copy.deepcopy(self.spec)
            bad[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                tuning.make_candidates(bad)
        bad = copy.deepcopy(self.spec)
        bad["anchors"][0]["learning_rate"] = 0.002
        with self.assertRaises(ValueError):
            tuning.make_candidates(bad)

    def test_train_only_normalization_and_no_test_access_through_selection(self):
        study = self.init()
        test_indices = np.asarray(study["split"]["test"])
        original_prepare = physics.prepare_dataset
        calls = []

        def guarded_prepare(arrays, indices, stats, fraction):
            self.assertFalse(np.intersect1d(indices, test_indices).size, "Tuning touched held-out test")
            calls.append(indices.copy())
            return original_prepare(arrays, indices, stats, fraction)

        with mock.patch.object(physics, "prepare_dataset", side_effect=guarded_prepare), \
                mock.patch.object(tuning, "fresh_holdout", side_effect=AssertionError("fresh test opened")), \
                mock.patch.object(legacy, "train_and_evaluate_run", side_effect=AssertionError("legacy auto-test called")), \
                contextlib.redirect_stdout(io.StringIO()):
            selection = tuning.tune(self.directory, study, self.cpu)
        self.assertEqual(len(calls), 2)
        self.assertIs(selection["test_used_for_selection"], False)
        self.assertFalse((self.directory / "final_evaluation_started.json").exists())
        self.assertFalse((self.directory / "final").exists())
        self.assertEqual(len(list((self.directory / "runs").glob("*/result.json"))), 8)
        arrays = physics.load_dataset(self.data_path)
        expected = physics.calculate_normalization_stats(arrays, np.asarray(study["split"]["train"])).to_dict()
        self.assertEqual(study["normalization"], expected)
        # Every sample outside training can be changed without changing fitted statistics.
        arrays.target[np.r_[study["split"]["validation"], study["split"]["test"]]] *= 2
        actual = physics.calculate_normalization_stats(arrays, np.asarray(study["split"]["train"])).to_dict()
        self.assertEqual(expected, actual)

    def test_finalization_gate_end_to_end_and_idempotence(self):
        study = self.init()
        with mock.patch.object(physics, "load_dataset", side_effect=AssertionError("test data opened")), \
                self.assertRaisesRegex(RuntimeError, "Test access denied"):
            tuning.finalize(self.directory, study, self.cpu)
        with contextlib.redirect_stdout(io.StringIO()):
            selection = tuning.tune(self.directory, study, self.cpu)
            final = tuning.finalize(self.directory, study, self.cpu)
            tuning.report(self.directory, study)
        candidates = 1 if selection["selected_candidate_id"] == "baseline" else 2
        self.assertEqual(len(final["records"]), candidates * 2 * 2 * 2 * 2)
        self.assertTrue((self.directory / "report.md").is_file())
        self.assertTrue((self.directory / "routing_comparisons.csv").is_file())
        fresh_meta = tuning.read_json(self.directory / "final" / "fresh_holdout.json")
        self.assertEqual(fresh_meta["original_sample_overlap"], 0)
        self.assertIs(fresh_meta["same_sensor_coordinates"], True)
        fresh = physics.load_dataset(self.directory / "final" / "fresh_holdout.npz")
        original = physics.load_dataset(self.data_path)
        np.testing.assert_array_equal(fresh.g05[0, :, :2], original.g05[0, :, :2])
        frozen = {p: legacy.file_sha256(self.directory / p) for p in selection["artifact_sha256"]}
        with mock.patch.object(physics, "load_dataset", side_effect=AssertionError("idempotence opened data")), \
                mock.patch.object(legacy, "run_epoch", side_effect=AssertionError("idempotence retrained")), \
                contextlib.redirect_stdout(io.StringIO()):
            tuning.tune(self.directory, study, self.cpu)
            self.assertEqual(final, tuning.finalize(self.directory, study, self.cpu))
        self.assertEqual(frozen, {p: legacy.file_sha256(self.directory / p) for p in frozen})
        data = tuning.development_data(study)[1.0]
        with self.assertRaisesRegex(RuntimeError, "locked"):
            self.run_trial(self.directory, self.config(study), data)

    def _resume_test(self, device):
        study = self.init(device=device)
        config = self.config(study)
        data = tuning.development_data(study)[1.0]
        continuous = self.directory / "continuous"
        interrupted = self.directory / "interrupted"
        self.run_trial(continuous, config, data, device)
        real_save = legacy.atomic_torch_save

        def fail_after_commit(value, path):
            real_save(value, path)
            if value.get("checkpoint_kind") == "latest" and value["epoch"] == 2:
                raise KeyboardInterrupt("simulated process death immediately after latest commit")

        with mock.patch.object(legacy, "atomic_torch_save", side_effect=fail_after_commit), \
                self.assertRaises(KeyboardInterrupt):
            self.run_trial(interrupted, config, data, device)
        status = tuning.read_json(tuning.trial_dir(interrupted, config) / "status.json")
        self.assertEqual(status["last_committed_epoch"], 2)
        resumed = self.run_trial(interrupted, config, data, device)
        self.assertTrue(resumed["training_result"]["resumed"])
        left = legacy.load_torch_checkpoint(tuning.trial_dir(continuous, config) / "latest.pt", self.cpu)
        right = legacy.load_torch_checkpoint(tuning.trial_dir(interrupted, config) / "latest.pt", self.cpu)
        for key in ("model_state_dict", "optimizer_state_dict", "rng_state", "shuffle_generator_state",
                    "history", "best_checkpoints", "early_stopping"):
            with self.subTest(state=key):
                self.assert_tree_equal(left[key], right[key])
        for objective in legacy.CHECKPOINT_SELECTIONS:
            model, stats, saved = tuning.load_trained_model(
                tuning.trial_dir(interrupted, config) / f"best_{objective}.pt", device)
            self.assertFalse(model.training)
            self.assertEqual(model.structure_dropout.p, 0.1)
            self.assertEqual(saved["checkpoint_selection"], objective)

    def test_resume_with_dropout_cpu_matches_uninterrupted(self):
        self._resume_test(self.cpu)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_resume_with_dropout_cuda_matches_uninterrupted(self):
        self._resume_test(torch.device("cuda"))

    def test_baseline_epoch_weights_optimizer_and_history_match_v10(self):
        study = self.init()
        config = self.config(study, candidate=0)
        data = tuning.development_data(study)[1.0]
        self.run_trial(self.directory, config, data)
        arrays = physics.load_dataset(self.data_path)
        test = physics.prepare_dataset(arrays, np.asarray(study["split"]["test"]),
                                       legacy.normalization_from_config(config), 1.0)
        reference = self.directory / "reference_v10"
        with contextlib.redirect_stdout(io.StringIO()):
            legacy.train_and_evaluate_run(run_config=config, datasets=(data.train, data.validation, test),
                                          experiment_results_dir=reference / "results",
                                          experiment_checkpoint_dir=reference / "checkpoints", device=self.cpu)
        a = legacy.load_torch_checkpoint(tuning.trial_dir(self.directory, config) / "latest.pt", self.cpu)
        b = legacy.load_torch_checkpoint(reference / "checkpoints" / legacy.run_id_for(config) / "latest.pt", self.cpu)
        for key in ("model_state_dict", "optimizer_state_dict", "history", "best_checkpoints", "shuffle_generator_state"):
            with self.subTest(state=key):
                self.assert_tree_equal(a[key], b[key])

    def test_incomplete_or_changed_trials_cannot_be_promoted(self):
        study = self.init()
        config = self.config(study)
        data = tuning.development_data(study)[1.0]
        self.run_trial(self.directory, config, data)
        with self.assertRaises(FileNotFoundError):
            tuning.promotion(self.directory, study)
        result_path = tuning.trial_dir(self.directory, config) / "result.json"
        record = tuning.read_json(result_path)
        record["evaluations"]["structure"]["selected_validation_loss"] -= 0.1
        legacy.atomic_write_json(result_path, record)
        with self.assertRaisesRegex(RuntimeError, "fingerprint"):
            tuning.read_trial(self.directory, config)

    def test_changed_split_and_environment_are_rejected(self):
        study = self.init()
        with mock.patch.object(tuning, "environment", return_value={"different": True}), \
                self.assertRaisesRegex(RuntimeError, "Runtime"):
            tuning.load_study(self.directory, device=self.cpu)
        changed = copy.deepcopy(study["split"])
        changed["validation"][0], changed["test"][0] = changed["test"][0], changed["validation"][0]
        legacy.atomic_save_npz(self.directory / "split_indices.npz", **{k: np.asarray(v) for k, v in changed.items()})
        with self.assertRaisesRegex(RuntimeError, "split changed"):
            tuning.load_study(self.directory)

    def test_source_drift_is_rejected_without_modifying_originals(self):
        self.init()
        real_hash = legacy.file_sha256

        def changed_source(path):
            if path == tuning.ROOT / "Codes" / "NewLearning9.py":
                return "changed"
            return real_hash(path)

        with mock.patch.object(legacy, "file_sha256", side_effect=changed_source), \
                self.assertRaisesRegex(RuntimeError, "Source changed"):
            tuning.load_study(self.directory)

    def test_frozen_checkpoint_change_is_rejected_before_opening_holdout(self):
        study = self.init()
        with contextlib.redirect_stdout(io.StringIO()):
            selected = tuning.tune(self.directory, study, self.cpu)
        relative = next(p for p in selected["artifact_sha256"] if p.endswith(".pt"))
        with (self.directory / relative).open("ab") as file:
            file.write(b"simulated corruption")
        with mock.patch.object(physics, "load_dataset", side_effect=AssertionError("test opened")), \
                self.assertRaisesRegex(RuntimeError, "frozen selection artifact"):
            tuning.finalize(self.directory, study, self.cpu)

    def test_resume_after_final_epoch_commit_does_not_train_again(self):
        study = self.init()
        config = self.config(study)
        data = tuning.development_data(study)[1.0]
        real_save = legacy.atomic_torch_save

        def interrupt_at_completion(value, path):
            real_save(value, path)
            if value.get("checkpoint_kind") == "latest" and value["epoch"] == self.spec["max_epochs"]:
                raise KeyboardInterrupt("committed final epoch")

        with mock.patch.object(legacy, "atomic_torch_save", side_effect=interrupt_at_completion), \
                self.assertRaises(KeyboardInterrupt):
            self.run_trial(self.directory, config, data)
        with mock.patch.object(legacy, "run_epoch", side_effect=AssertionError("retrained completed epoch")):
            record = self.run_trial(self.directory, config, data)
        self.assertEqual(record["training_result"]["epochs_completed"], 3)
        self.assertEqual(record["training_result"]["stop_reason"], "max_epochs")

    def test_cross_split_physical_duplicates_are_rejected(self):
        arrays = physics.load_dataset(self.data_path)
        split = physics.create_data_split(len(arrays.target))
        arrays.g00[split.test[0]] = arrays.g00[split.train[0]]
        with self.assertRaisesRegex(ValueError, "cross splits"):
            tuning.validate_split(arrays, split)

    def test_common_setting_guardrail_and_baseline_fallback(self):
        def row(name, score, a, b):
            return {"candidate_id": name, "score": score, "by_condition": {"sign": a, "full": b}}
        rows = [row("baseline", 1.0, 1.0, 1.0), row("harm_sign", 0.8, 1.1, 0.5),
                row("balanced", 0.9, 0.91, 0.89)]
        selected, _, _ = tuning.choose_setting(rows, self.default)
        self.assertEqual(selected, "balanced")
        selected, _, _ = tuning.choose_setting(rows[:2], self.default)
        self.assertEqual(selected, "baseline")
        selected, _, _ = tuning.choose_setting(
            [rows[0], row("tiny", 0.999, 0.999, 0.999)], self.default)
        self.assertEqual(selected, "baseline")


if __name__ == "__main__":
    unittest.main()
