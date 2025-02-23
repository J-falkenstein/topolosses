from __future__ import annotations

import os
import sys
import unittest

import numpy as np
import torch
from parameterized import parameterized

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from topolosses.losses.cldice.src.cldice_loss import BaseCLDiceLoss, DiceCLDiceLoss

# When pip package is available:
# from topolosses.losses import DiceCLDiceLoss, DiceCLDiceLoss

# TODO add more real test cases (multichannel, inlcude_background=False, weights, etc.)
TEST_CASES = [
    [  # shape: (1, 1, 2, 2)
        {"include_background": True, "sigmoid": True, "smooth": 1e-6, "alpha": 0},
        {"input": torch.tensor([[[[1.0, -1.0], [-1.0, 1.0]]]]), "target": torch.tensor([[[[1.0, 0.0], [1.0, 1.0]]]])},
        0.307576,
    ],
    [  # shape: (2, 1, 2, 2)
        {"include_background": True, "sigmoid": True, "smooth": 1e-4, "alpha": 0},
        {
            "input": torch.tensor([[[[1.0, -1.0], [-1.0, 1.0]]], [[[1.0, -1.0], [-1.0, 1.0]]]]),
            "target": torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]], [[[1.0, 0.0], [1.0, 0.0]]]]),
        },
        0.416657,
    ],
    [  # shape: (2, 1, 2, 2)
        {"include_background": True, "smooth": 1e-4, "alpha": 0},
        {
            "input": torch.tensor([[[[0.3, 0.4], [0.7, 0.9]]], [[[1.0, 0.1], [0.5, 0.3]]]]),
            "target": torch.tensor([[[[0.3, 0.4], [0.7, 0.9]]], [[[1.0, 0.1], [0.5, 0.3]]]]),
        },
        0.307773,
    ],
    [  # shape: (2, 1, 2, 2)
        {"include_background": True, "smooth": 1e-4, "alpha": 0, "weights": torch.tensor([0.5])},
        {
            "input": torch.tensor([[[[0.3, 0.4], [0.7, 0.9]]], [[[1.0, 0.1], [0.5, 0.3]]]]),
            "target": torch.tensor([[[[0.3, 0.4], [0.7, 0.9]]], [[[1.0, 0.1], [0.5, 0.3]]]]),
        },
        0.307773,
    ],
    [  # shape: (2, 1, 2, 2)
        {"include_background": True, "smooth": 1e-4, "alpha": 0, "weights": torch.tensor([8])},
        {
            "input": torch.tensor([[[[0.3, 0.4], [0.7, 0.9]]], [[[1.0, 0.1], [0.5, 0.3]]]]),
            "target": torch.tensor([[[[0.3, 0.4], [0.7, 0.9]]], [[[1.0, 0.1], [0.5, 0.3]]]]),
        },
        0.307773,
    ],
    [  # shape: (1, 2, 6, 6,)
        {},
        {
            "input": torch.nn.functional.one_hot(
                torch.tensor(
                    [
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        ]
                    ]
                ).to(torch.int64),
                2,
            )
            .permute(0, 3, 1, 2)
            .float(),
            "target": torch.nn.functional.one_hot(
                torch.tensor(
                    [
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        ]
                    ]
                ).to(torch.int64),
                2,
            )
            .permute(0, 3, 1, 2)
            .float(),
        },
        0.3329663,
    ],
    [  # shape: (1, 2, 6, 6,)
        {},
        {
            "input": torch.nn.functional.one_hot(
                1
                - torch.tensor(
                    [
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        ]
                    ]
                ).to(torch.int64),
                2,
            )
            .permute(0, 3, 1, 2)
            .float(),
            "target": torch.nn.functional.one_hot(
                1
                - torch.tensor(
                    [
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        ]
                    ]
                ).to(torch.int64),
                2,
            )
            .permute(0, 3, 1, 2)
            .float(),
        },
        0.19011,
    ],
]


class TestDiceCLDiceLoss(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_result_DiceCLDice(self, input_param, input_data, expected_val):
        result = DiceCLDiceLoss(**input_param).forward(**input_data)
        np.testing.assert_allclose(result.detach().cpu().numpy(), expected_val, rtol=1e-5)

    # TODO add Test cases for BaseCLDiceLoss
    # def test_result_BaseCLDice(self, input_param, input_data, expected_val):
    #     result = DiceCLDiceLoss(**input_param).forward(**input_data)
    #     np.testing.assert_allclose(result.detach().cpu().numpy(), expected_val, rtol=1e-5)

    def test_with_cuda(self):
        if torch.cuda.is_available():
            loss = DiceCLDiceLoss().cuda()
            input_data = {
                "input": torch.tensor([[[[0.3, 0.4], [0.7, 0.9]]], [[[1.0, 0.1], [0.5, 0.3]]]]).cuda(),
                "target": torch.tensor([[[[0.3, 0.4], [0.7, 0.9]]], [[[1.0, 0.1], [0.5, 0.3]]]]).cuda(),
            }
            result = loss.forward(**input_data)
            np.testing.assert_allclose(result.detach().cpu().numpy(), 0.307773, rtol=1e-5)

    def test_ill_shape(self):
        loss = DiceCLDiceLoss()
        with self.assertRaisesRegex(ValueError, ""):
            loss.forward(torch.ones((1, 2, 3)), torch.ones((4, 5, 6)))

    def test_ill_opts(self):
        with self.assertRaisesRegex(ValueError, ""):
            DiceCLDiceLoss(sigmoid=True, softmax=True)
        chn_input = torch.ones((1, 1, 3, 3))
        chn_target = torch.ones((1, 1, 3, 3))
        with self.assertRaisesRegex(TypeError, ""):
            DiceCLDiceLoss(include_background=True, batch="unknown")(chn_input, chn_target)
        with self.assertRaisesRegex(ValueError, ""):
            loss = DiceCLDiceLoss(softmax=True)
            loss.forward(chn_input, chn_target)
        chn_input = torch.ones((1, 1, 3))
        chn_target = torch.ones((1, 1, 3))
        with self.assertRaisesRegex(ValueError, ""):
            loss = DiceCLDiceLoss()
            loss.forward(chn_input, chn_target)

    def test_input_warnings(self):
        chn_input = torch.ones((1, 1, 3, 3))
        chn_target = torch.ones((1, 1, 3, 3))
        with self.assertWarns(Warning):
            loss = DiceCLDiceLoss(include_background=False)
            loss.forward(chn_input, chn_target)

    # from test_utils import test_script_save
    # def test_script(self):
    #     loss = DiceCLDiceLoss()
    #     test_input = torch.ones(2, 1, 8, 8)
    #     test_script_save(loss, test_input, test_input)


if __name__ == "__main__":
    unittest.main()
