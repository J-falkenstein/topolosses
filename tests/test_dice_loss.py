from __future__ import annotations

import os
import sys
import unittest

import numpy as np
import torch
from parameterized import parameterized

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# from topolosses.losses.dice.src.dice_loss import DiceLoss

# When pip package is available_
from topolosses.losses import DiceLoss

TEST_CASES = [
    [  # shape: (1, 1, 2, 2), (1, 1, 2, 2)
        {"sigmoid": True, "smooth": 1e-6},
        {"input": torch.tensor([[[[1.0, -1.0], [-1.0, 1.0]]]]), "target": torch.tensor([[[[1.0, 0.0], [1.0, 1.0]]]])},
        0.307576,
    ],
    [  # shape: (2, 1, 2, 2), (2, 1, 2, 2)
        {"sigmoid": True, "smooth": 1e-4},
        {
            "input": torch.tensor([[[[1.0, -1.0], [-1.0, 1.0]]], [[[1.0, -1.0], [-1.0, 1.0]]]]),
            "target": torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]], [[[1.0, 0.0], [1.0, 0.0]]]]),
        },
        0.416657,
    ],
    [  # shape: (2, 1, 2, 2), (2, 1, 2, 2)
        {"smooth": 1e-4},
        {
            "input": torch.tensor([[[[0.3, 0.4], [0.7, 0.9]]], [[[1.0, 0.1], [0.5, 0.3]]]]),
            "target": torch.tensor([[[[0.3, 0.4], [0.7, 0.9]]], [[[1.0, 0.1], [0.5, 0.3]]]]),
        },
        0.307773,
    ],
    [  # shape: (2, 2, 3), (2, 2, 3) - one dimensional spatial data
        {"include_background": True, "softmax": True, "smooth": 1e-4},
        {
            "input": torch.tensor([[[-1.0, 0.0, 1.0], [1.0, 0.0, -1.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]),
            "target": torch.nn.functional.one_hot(torch.tensor([[[1.0, 0.0, 0.0]], [[1.0, 1.0, 0.0]]]).to(torch.int64))
            .permute(0, 3, 2, 1)
            .squeeze(-1),
        },
        0.383713,
    ],
    [  # shape: (2, 2, 3), (2, 2, 3)
        {"include_background": False, "smooth": 0},
        {
            "input": torch.tensor([[[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]]),
            "target": torch.nn.functional.one_hot(torch.tensor([[[0.0, 0.0, 1.0]], [[0.0, 1.0, 0.0]]]).to(torch.int64))
            .permute(0, 3, 2, 1)
            .squeeze(-1),
        },
        0.0,
    ],
    [  # shape: (2, 2, 3), (2, 2, 3)
        {"weights": torch.tensor([0.0, 1.0]), "smooth": 0},
        {
            "input": torch.tensor([[[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]]),
            "target": torch.nn.functional.one_hot(torch.tensor([[[0.0, 0.0, 1.0]], [[0.0, 1.0, 0.0]]]).to(torch.int64))
            .permute(0, 3, 2, 1)
            .squeeze(-1),
        },
        0.0,
    ],
]


class TestDiceLoss(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_result(self, input_param, input_data, expected_val):
        result = DiceLoss(**input_param).forward(**input_data)
        np.testing.assert_allclose(result.detach().cpu().numpy(), expected_val, rtol=1e-5)

    def test_with_cuda(self):
        if torch.cuda.is_available():
            loss = DiceLoss(smooth=1e-4).cuda()
            input_data = {
                "input": torch.tensor([[[[0.3, 0.4], [0.7, 0.9]]], [[[1.0, 0.1], [0.5, 0.3]]]]).cuda(),
                "target": torch.tensor([[[[0.3, 0.4], [0.7, 0.9]]], [[[1.0, 0.1], [0.5, 0.3]]]]).cuda(),
            }
            result = loss.forward(**input_data)
            np.testing.assert_allclose(result.detach().cpu().numpy(), 0.307773, rtol=1e-5)

    def test_ill_shape(self):
        loss = DiceLoss()
        with self.assertRaisesRegex(ValueError, ""):
            loss.forward(torch.ones((1, 2, 3)), torch.ones((4, 5, 6)))

    def test_ill_opts(self):
        with self.assertRaisesRegex(ValueError, ""):
            DiceLoss(sigmoid=True, softmax=True)
        chn_input = torch.ones((1, 1, 3))
        chn_target = torch.ones((1, 1, 3))
        with self.assertRaisesRegex(TypeError, ""):
            DiceLoss(batch="unknown")(chn_input, chn_target)
        with self.assertRaisesRegex(ValueError, ""):
            loss = DiceLoss(softmax=True)
            loss.forward(chn_input, chn_target)

    def test_input_warnings(self):
        chn_input = torch.ones((1, 1, 3))
        chn_target = torch.ones((1, 1, 3))
        with self.assertWarns(Warning):
            loss = DiceLoss(include_background=False)
            loss.forward(chn_input, chn_target)

    # from test_utils import test_script_save
    # def test_script(self):
    #     loss = DiceLoss()
    #     test_input = torch.ones(2, 1, 8, 8)
    #     test_script_save(loss, test_input, test_input)


if __name__ == "__main__":
    unittest.main()
