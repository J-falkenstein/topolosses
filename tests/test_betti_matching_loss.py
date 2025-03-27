from __future__ import annotations

import os
import sys
import unittest

import numpy as np
import torch
import torch.nn.functional as F
from parameterized import parameterized

# sys.path needs to be added if using local implementations otherwise looking for the package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from topolosses.losses import DiceLoss, CLDiceLoss, BettiMatchingLoss

from losses_original.betti_losses import (
    FastMulticlassBettiMatchingLoss,
    FastMulticlassDiceBettiMatchingLoss,
)


def transform(tensor, num_classes=2):
    return F.one_hot(tensor.to(torch.int64), num_classes).permute(0, 3, 1, 2).float()


# TODO make betti matching loss test cases
TEST_CASES = [
    [
        # shape [1, 5, 5]
        {"use_base_loss": False, "alpha": 1, "include_background": True},
        {
            "input": transform(
                torch.tensor(
                    [
                        [
                            [1.0, 1.0, 1.0, 1.0, 1.0],
                            [1.0, 0.0, 0.0, 0.0, 1.0],
                            [1.0, 1.0, 1.0, 1.0, 1.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ]
                    ]
                )
            ),
            "target": transform(
                torch.tensor(
                    [
                        [
                            [0.0, 0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ]
                    ]
                )
            ),
        },
        3,
    ],
    [
        {"use_base_loss": False, "alpha": 1},
        {
            "input": transform(
                torch.tensor(
                    [
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ]
                    ]
                )
            ),
            "target": transform(
                torch.tensor(
                    [
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 1.0, 1.0, 0.0],
                            [0.0, 1.0, 0.0, 1.0, 0.0],
                            [0.0, 1.0, 1.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ]
                    ]
                )
            ),
        },
        0,
    ],
    [
        {"use_base_loss": False, "alpha": 1, "include_background": True},
        {
            "input": transform(
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
                )
            ),
            "target": transform(
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
                )
            ),
        },
        1,
    ],
    [
        {"use_base_loss": False, "alpha": 1, "include_background": True},
        {
            "input": transform(
                torch.tensor(
                    [
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ]
                    ]
                )
            ),
            "target": transform(
                torch.tensor(
                    [
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ]
                    ]
                )
            ),
        },
        0,
    ],
    [
        {"use_base_loss": False, "alpha": 1, "include_background": True},
        {
            "input": transform(
                torch.tensor(
                    [
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                            [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                            [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        ]
                    ]
                )
            ),
            "target": transform(
                torch.tensor(
                    [
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                            [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        ]
                    ]
                )
            ),
        },
        0,
    ],
]

TEST_CASES_OLDNEW = [
    [
        {},
        {
            "input": transform(
                torch.tensor(
                    [
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ]
                    ]
                )
            ),
            "target": transform(
                torch.tensor(
                    [
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 1.0, 1.0, 0.0],
                            [0.0, 1.0, 0.0, 1.0, 0.0],
                            [0.0, 1.0, 1.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ]
                    ]
                )
            ),
        },
        0,
    ]
]


class TestDiceTopographLoss(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_result(self, input_param, input_data, expected_val):
        result_original = FastMulticlassBettiMatchingLoss(ignore_background=True).forward(
            prediction=input_data["input"], target=input_data["target"]
        )
        result = BettiMatchingLoss(**input_param).forward(**input_data)
        # TODO change to compare to expected val
        np.testing.assert_allclose(result.detach().cpu().numpy(), result_original.detach().cpu().numpy(), rtol=1e-5)

    @parameterized.expand(TEST_CASES_OLDNEW)
    def test_compare_old_to_new(self, input_param, input_data, expected_val):
        # settings as defined in old betti loss implementation
        clDiceLoss = CLDiceLoss(
            softmax=True,
            include_background=True,
            smooth=1e-5,
            alpha=0.5,
            iter_=5,
            batch=True,
            base_loss=DiceLoss(
                softmax=True,
                smooth=1e-5,
                batch=True,
            ),
        )
        result = BettiMatchingLoss(**input_param, alpha=0.5, softmax=True, base_loss=clDiceLoss).forward(**input_data)
        print("new implementation: ")
        print(result)

        result_original = FastMulticlassDiceBettiMatchingLoss(ignore_background=True).forward(
            prediction=input_data["input"], target=input_data["target"]
        )
        print("old implementation: ")
        print(
            result_original
        )  # difference stem from the fact that the old implementation defines the cl dice different by multiplying (1-alph)*dice
        # np.testing.assert_allclose(result.detach().cpu().numpy(), result_original.detach().cpu().numpy(), rtol=1e-5)

    def test_with_cuda(self):
        if torch.cuda.is_available():
            loss = BettiMatchingLoss(include_background=True).cuda()
            input_data = {
                "input": torch.tensor([[[[0.3, 0.4], [0.7, 0.9]]], [[[1.0, 0.1], [0.5, 0.3]]]]).cuda(),
                "target": torch.tensor([[[[0.3, 0.4], [0.7, 0.9]]], [[[1.0, 0.1], [0.5, 0.3]]]]).cuda(),
            }
            result = loss.forward(**input_data)
            np.testing.assert_allclose(result.detach().cpu().numpy(), 0.307773, rtol=1e-4)

    def test_ill_shape(self):
        loss = BettiMatchingLoss()
        with self.assertRaisesRegex(ValueError, ""):
            loss.forward(torch.ones((1, 2, 3)), torch.ones((4, 5, 6)))

    def test_ill_opts(self):
        with self.assertRaisesRegex(ValueError, ""):
            BettiMatchingLoss(sigmoid=True, softmax=True)
        chn_input = torch.ones((1, 1, 3, 3))
        chn_target = torch.ones((1, 1, 3, 3))
        with self.assertRaisesRegex(ValueError, ""):
            loss = BettiMatchingLoss(softmax=True)
            loss.forward(chn_input, chn_target)

    def test_input_warnings(self):
        chn_input = torch.ones((1, 1, 3, 3))
        chn_target = torch.ones((1, 1, 3, 3))
        with self.assertWarns(Warning):
            loss = BettiMatchingLoss(include_background=False)
            loss.forward(chn_input, chn_target)
        with self.assertWarns(Warning):
            loss = BettiMatchingLoss(use_base_loss=False, alpha=0.5)
        with self.assertWarns(Warning):
            loss = BettiMatchingLoss(use_base_loss=False, base_loss=DiceLoss())


if __name__ == "__main__":
    unittest.main()
