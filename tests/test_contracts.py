"""Small CPU checks for the paper equations and the mask-file interface."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np
import torch

from src.flowanchor.ptp_utils import CAProcessor
from src.flowanchor.utils import get_diff_mask
from src.wanedit_mask.pipeline import FlowEditWan

ROOT = Path(__file__).resolve().parents[1]


class ToyScheduler:
    def set_timesteps(self, steps, device):
        self.timesteps = torch.linspace(1000, 1000 / steps, steps, device=device)

    def _init_step_index(self, timestep):
        pass


class ToyTransformer:
    def __call__(self, hidden_states, encoder_hidden_states, **kwargs):
        velocity = torch.zeros_like(hidden_states)
        if encoder_hidden_states == "target":
            velocity[..., 0] = 2
            velocity[..., 1] = 4
        return (velocity,)


class ToyPipeline:
    transformer = ToyTransformer()

    def encode_prompt(self, prompt, **kwargs):
        return prompt, "negative"


class EquationContracts(unittest.TestCase):
    def test_wanedit_updates_only_masked_velocity(self):
        source = torch.tensor([3.0, 7.0]).reshape(1, 1, 1, 1, 2)
        mask = torch.tensor([1.0, 0.0]).reshape_as(source)
        output = FlowEditWan(
            ToyPipeline(),
            ToyScheduler(),
            source,
            "source",
            "target",
            "negative",
            T_steps=2,
            n_max=2,
            tar_guidance_scale=1,
            mask=mask,
        )
        # Integrated time is -1: editable value 3-2=1; background stays 7.
        torch.testing.assert_close(output.flatten(), torch.tensor([1.0, 7.0]))

    def test_sar_matches_hand_calculated_two_stage_example(self):
        logits = torch.tensor([[[[0.0, 1.0, 2.0], [3.0, 2.0, 0.0]]]])
        config = dict(
            ca_mask=torch.tensor([1.0, 0.0]).reshape(1, 1, 1, 1, 2),
            target_word_ids=[1],
            row_strength=0.3,
            col_strength=0.3,
        )
        output = CAProcessor(None, 0, config).modulate_attention_weights(logits)
        expected = torch.tensor([[[[0.0, 1.51, 1.4], [3.0, 1.79, 0.0]]]])
        torch.testing.assert_close(output, expected, atol=1e-6, rtol=0)

    def test_amm_uses_signed_channel_mean_and_sample_normalization(self):
        signal = torch.tensor([[-2.0, 2.0], [0.0, 4.0]]).reshape(1, 2, 1, 1, 2)
        batch = torch.cat([signal, 10 * signal + 30, torch.ones_like(signal)])
        output = get_diff_mask(batch).flatten(1)
        torch.testing.assert_close(output[0], torch.tensor([0.0, 1.0]))
        torch.testing.assert_close(output[1], output[0])
        torch.testing.assert_close(output[2], torch.zeros(2))


class MaskFileContracts(unittest.TestCase):
    def make_video(self, path, frames, fps=8):
        height, width = frames[0].shape[:2]
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        self.assertTrue(writer.isOpened())
        for frame in frames:
            writer.write(frame)
        writer.release()

    def run_converter(self, source, mask, output):
        return subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/mask_video_to_frames.py"),
                "--source_video",
                str(source),
                "--mask_video",
                str(mask),
                "--output_dir",
                str(output),
            ],
            capture_output=True,
            text=True,
        )

    def test_mask_video_exports_binary_numbered_frames(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            frame = np.zeros((16, 16, 3), dtype=np.uint8)
            frame[:, 8:] = 255
            self.make_video(root / "source.mp4", [frame] * 3)
            self.make_video(root / "mask.mp4", [frame] * 3)
            result = self.run_converter(
                root / "source.mp4", root / "mask.mp4", root / "masks"
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            files = sorted((root / "masks").glob("*.png"))
            self.assertEqual(
                [p.name for p in files], [f"{i:05d}.png" for i in range(3)]
            )
            for path in files:
                actual = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                np.testing.assert_array_equal(actual, frame[:, :, 0])

    def test_misaligned_mask_sequence_is_rejected_without_png_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            frame = np.zeros((16, 16, 3), dtype=np.uint8)
            self.make_video(root / "source.mp4", [frame] * 3)
            self.make_video(root / "mask.mp4", [frame] * 2)
            result = self.run_converter(
                root / "source.mp4", root / "mask.mp4", root / "masks"
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("frame counts differ", result.stderr)
            self.assertFalse((root / "masks").exists())


if __name__ == "__main__":
    unittest.main()
