from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

import ModelExperiment as experiment


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


if __name__ == "__main__":
    unittest.main()
