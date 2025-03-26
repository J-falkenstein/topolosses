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
from topolosses.losses import BettiMatchingLoss
from topolosses.losses import DiceLoss

from losses_original.betti_losses import FastBettiMatchingLoss


def transform(tensor, num_classes=2):
    return F.one_hot(tensor.to(torch.int64), num_classes).permute(0, 3, 1, 2).float()


# TODO make betti matching loss test cases
TEST_CASES = [
    [
        # shape [1, 5, 5]
        {"use_base_loss": False, "alpha": 1},
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
        # shape [1, 6, 6]
        {"use_base_loss": False, "alpha": 1},
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
        {"use_base_loss": False, "alpha": 1},
        {
            "input": transform(
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
                )
            ),
            "target": transform(
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
                )
            ),
        },
        1,
    ],
    [
        {"use_base_loss": False, "alpha": 1},
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
        {"use_base_loss": False, "alpha": 1},
        {
            "input": transform(
                torch.tensor(
                    [
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 1.0, 1.0, 0.0],
                            [0.0, 1.0, 1.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0, 0.0],
                        ]
                    ]
                )
            ),
            "target": transform(
                torch.tensor(
                    [
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 1.0, 0.0, 0.0],
                            [0.0, 1.0, 1.0, 0.0, 0.0],
                            [0.0, 1.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ]
                    ]
                )
            ),
        },
        0,
    ],
    [
        {"use_base_loss": False, "alpha": 1},
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
    [
        {"use_base_loss": False, "alpha": 1},
        {
            "input": transform(
                torch.tensor(
                    [
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                            [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                            [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
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
                            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                        ]
                    ]
                )
            ),
        },
        1,
    ],
]


# TODO use betti matching
class TestDiceTopographLoss(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_result_DiceCLDice(self, input_param, input_data, expected_val):
        result = BettiMatchingLoss(**input_param).forward(**input_data)
        result_orignal = FastBettiMatchingLoss(**input_param).forward(**input_data)
        print("comparing old with new")
        print(result)
        print(result_orignal)
        # np.testing.assert_allclose(result.detach().cpu().numpy(), expected_val, rtol=1e-5)

    # @parameterized.expand(TEST_CASES)
    # def test_result_CLDice(self, input_param, input_data, expected_val):
    #     result = TopographLoss(**input_param).forward(**input_data)
    #     print(result.item())
    #     expected_val = cldice_loss_orignal.Multiclass_CLDice(**input_param).forward(**input_data)
    #     print(expected_val[0].item())
    #     np.testing.assert_allclose(result.detach().cpu().numpy(), expected_val[0].item(), rtol=1e-5)

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

    # from test_utils import test_script_save
    # def test_script(self):
    #     loss = DiceTopographLoss()
    #     test_input = torch.ones(2, 1, 8, 8)
    #     test_script_save(loss, test_input, test_input)


if __name__ == "__main__":
    unittest.main()
