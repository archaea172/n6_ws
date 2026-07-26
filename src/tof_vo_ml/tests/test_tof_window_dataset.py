from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from tof_vo_ml.dataset import ToFWindowDataset
from tof_vo_ml.model import ToFOdometryCNN


class ToFOdometryCNNTest(unittest.TestCase):
    def test_output_shape(self):
        model = ToFOdometryCNN(input_channels=8, output_dim=3)
        output = model(torch.randn(2, 8, 8, 8))

        self.assertEqual(tuple(output.shape), (2, 3))


class ToFWindowDatasetTest(unittest.TestCase):
    def test_uses_8_frame_window_and_final_transition_label(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            npz_path = Path(temp_dir) / "sample.npz"
            frame_count = 10
            distance = (
                np.arange(frame_count * 8 * 8, dtype=np.float32)
                .reshape(frame_count, 8, 8)
                % 4.0
            )
            valid = np.ones_like(distance, dtype=bool)
            valid[0, 0, 0] = False
            pose_xytheta = np.zeros((frame_count, 3), dtype=np.float32)
            pose_xytheta[:, 0] = np.arange(frame_count, dtype=np.float32) * 0.1

            np.savez_compressed(
                npz_path,
                distance_m=distance,
                valid=valid,
                pose_xytheta=pose_xytheta,
            )

            dataset = ToFWindowDataset(npz_path, window_size=8, max_distance_m=4.0)
            inputs, label = dataset[0]

            self.assertEqual(len(dataset), frame_count - 7)
            self.assertEqual(tuple(inputs.shape), (8, 8, 8))
            self.assertEqual(float(inputs[0, 0, 0]), 0.0)
            self.assertTrue(
                np.allclose(
                    label.numpy(),
                    np.asarray([0.1, 0.0, 0.0], dtype=np.float32),
                )
            )


if __name__ == "__main__":
    unittest.main()
