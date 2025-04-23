from __future__ import annotations

import os
import sys
import unittest

import numpy as np
import torch
import torch.nn.functional as F
from parameterized import parameterized

from losses_original.utils import convert_to_one_vs_rest

# sys.path needs to be added if using local implementations otherwise looking for the package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from topolosses.losses import WarpingLoss, DiceLoss, CLDiceLoss


def transform(tensor, num_classes=2):
    return F.one_hot(tensor.to(torch.int64), num_classes).permute(0, 3, 1, 2).float()


TEST_CASES = [
    [
        # shape [1, 5, 5]
        {
            "use_base_loss": False,
            "alpha": 1,
        },
        {
            "input": convert_to_one_vs_rest(
                transform(
                    torch.cat(
                        [
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
                            ),
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
                            ),
                        ]
                    )
                )
            ),
            "target": convert_to_one_vs_rest(
                transform(
                    torch.cat(
                        [
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
                            ),
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
                            ),
                        ]
                    )
                )
            ),
        },
        2.131094026,
    ],
    [
        {
            "use_base_loss": False,
            "alpha": 1,
        },
        {
            "input": transform(
                torch.tensor(
                    [
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 1.0, 0.0],
                            [1.0, 0.0, 1.0, 0.0, 0.0],
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
                            [0.0, 1.0, 0.0, 1.0, 1.0],
                            [0.0, 1.0, 1.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ]
                    ]
                )
            ),
        },
        3.4933026175447077,
    ],
    [
        {
            "use_base_loss": False,
            "alpha": 1,
        },
        {
            "input": transform(
                torch.tensor(
                    [
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                        ]
                    ]
                )
            ),
            "target": transform(
                torch.tensor(
                    [
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
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
        3.6297278090755296,
    ],
    [
        {
            "use_base_loss": False,
            "alpha": 1,
        },
        {
            "input": transform(
                torch.tensor(
                    [
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
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
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 1.0, 1.0, 0.0],
                            [0.0, 1.0, 0.0, 1.0, 0.0],
                            [0.0, 1.0, 1.0, 1.0, 0.0],
                        ]
                    ]
                )
            ),
        },
        3.270061395039729,
    ],
    [
        {
            "use_base_loss": False,
            "alpha": 1,
        },
        {
            "input": transform(
                torch.tensor(
                    [
                        [
                            [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                            [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                            [0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0],
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
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                        ]
                    ]
                )
            ),
        },
        3.452552,
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
        1.91770518346,
    ]
]


class TestLoss(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_result(self, input_param, input_data, expected_val):
        result = WarpingLoss(**input_param).forward(**input_data)
        np.testing.assert_allclose(result.detach().cpu().numpy(), expected_val, rtol=1e-5)

    @parameterized.expand(TEST_CASES_OLDNEW)
    def test_compare_old_to_new(self, input_param, input_data, expected_val):
        # settings as defined in old betti loss implementation
        clDiceLoss = CLDiceLoss(
            softmax=True,
            include_background=False,
            smooth=1e-5,
            alpha=0.5,
            iter_=3,
            batch=True,
            base_loss=DiceLoss(
                softmax=True,
                smooth=1e-5,
                batch=True,
            ),
        )
        result = WarpingLoss(**input_param, alpha=0.5, softmax=True, base_loss=clDiceLoss).forward(**input_data)
        np.testing.assert_allclose(result.detach().cpu().numpy(), expected_val, rtol=1e-5)

    def test_with_cuda(self):
        if torch.cuda.is_available():
            loss = WarpingLoss().cuda()
            input_data = {
                "input": torch.tensor(
                    [
                        [[[0.3, 0.4], [0.7, 0.9]], [[0.7, 0.6], [0.3, 0.1]]],
                        [[[1.0, 0.1], [0.5, 0.3]], [[0.0, 0.9], [0.5, 0.7]]],
                    ]
                ).cuda(),
                "target": torch.tensor(
                    [
                        [[[0.3, 0.4], [0.7, 0.9]], [[0.7, 0.6], [0.3, 0.1]]],
                        [[[1.0, 0.1], [0.5, 0.3]], [[0.0, 0.9], [0.5, 0.7]]],
                    ]
                ).cuda(),
            }
            result = loss.forward(**input_data)
            np.testing.assert_allclose(result.detach().cpu().numpy(), 1.58324613, rtol=1e-4)

    def test_ill_shape(self):
        loss = WarpingLoss()
        with self.assertRaisesRegex(ValueError, ""):
            loss.forward(torch.ones((1, 2, 3)), torch.ones((4, 5, 6)))

    def test_ill_opts(self):
        with self.assertRaisesRegex(ValueError, ""):
            WarpingLoss(sigmoid=True, softmax=True)
        chn_input = torch.ones((1, 1, 3, 3))
        chn_target = torch.ones((1, 1, 3, 3))
        with self.assertRaisesRegex(ValueError, ""):
            loss = WarpingLoss()
            loss.forward(chn_input, chn_target)

    def test_input_warnings(self):
        with self.assertWarns(Warning):
            loss = WarpingLoss(use_base_loss=False, alpha=0.5)
        with self.assertWarns(Warning):
            loss = WarpingLoss(use_base_loss=False, base_loss=DiceLoss())


if __name__ == "__main__":
    unittest.main()
