from __future__ import annotations

import os
import sys
import unittest

import numpy as np
import torch
import torch.nn.functional as F
from parameterized import parameterized

from losses_original.utils import convert_to_one_vs_rest

# from losses_original.mosin import MulticlassMOSIN, MulticlassDiceMOSIN

# sys.path needs to be added if using local implementations otherwise looking for the package
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from topolosses.losses import MosinLoss, DiceLoss, CLDiceLoss


def transform(tensor, num_classes=2):
    return F.one_hot(tensor.to(torch.int64), num_classes).permute(0, 3, 1, 2).float()


def generate_random_binary_tensor(batch=1, height=32, width=32, prob_one=0.4, seed=1):
    torch.manual_seed(seed)
    random_tensor = torch.rand(batch, height, width)
    binary_tensor = (random_tensor < prob_one).float()
    return binary_tensor


TEST_CASES = [
    [
        {"use_base_loss": False, "alpha": 1, "include_background": True},
        {
            "input": convert_to_one_vs_rest(
                transform(generate_random_binary_tensor(height=32, width=32, prob_one=0.25, seed=1))
            ),
            "target": convert_to_one_vs_rest(
                transform(generate_random_binary_tensor(height=32, width=32, prob_one=0.25, seed=1))
            ),
        },
        6.84917736,
    ],
    [
        {"use_base_loss": False, "alpha": 1, "include_background": True},
        {
            "input": transform(generate_random_binary_tensor(height=32, width=32, prob_one=0.5, seed=2)),
            "target": transform(generate_random_binary_tensor(height=32, width=32, prob_one=0.5, seed=2)),
        },
        18.4565678,
    ],
    [
        {"use_base_loss": False, "alpha": 1, "include_background": True},
        {
            "input": transform(generate_random_binary_tensor(height=32, width=32, prob_one=0.75, seed=3)),
            "target": transform(generate_random_binary_tensor(height=32, width=32, prob_one=0.75, seed=3)),
        },
        15.17680645,
    ],
]

TEST_CASES_OLDNEW = [
    [
        {},
        {
            "input": transform(generate_random_binary_tensor(height=32, width=32, prob_one=0.75, seed=3)),
            "target": transform(generate_random_binary_tensor(height=32, width=32, prob_one=0.75, seed=3)),
        },
        3.903480768,
    ],
    [
        {},
        {
            "input": transform(generate_random_binary_tensor(batch=4, height=128, width=128, prob_one=0.7, seed=3)),
            "target": transform(generate_random_binary_tensor(batch=4, height=128, width=128, prob_one=0.7, seed=3)),
        },
        2.537840127,
    ],
]


class TestDiceTopographLoss(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_result(self, input_param, input_data, expected_val):
        result = MosinLoss(**input_param)(**input_data)
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
        result = MosinLoss(
            **input_param, alpha=0.5, softmax=True, base_loss=clDiceLoss, include_background=True
        ).forward(**input_data)
        np.testing.assert_allclose(result.detach().cpu().numpy(), expected_val, rtol=1e-5)

    def test_with_cuda(self):
        if torch.cuda.is_available():
            loss = MosinLoss(include_background=True).cuda()
            input_data = {
                "input": transform(generate_random_binary_tensor(height=32, width=32, prob_one=0.75, seed=3)).cuda(),
                "target": transform(generate_random_binary_tensor(height=32, width=32, prob_one=0.75, seed=3)).cuda(),
            }
            result = loss.forward(**input_data)
            np.testing.assert_allclose(result.detach().cpu().numpy(), 7.587616, rtol=1e-4)

    def test_ill_shape(self):
        loss = MosinLoss()
        with self.assertRaisesRegex(ValueError, ""):
            loss.forward(torch.ones((1, 2, 3)), torch.ones((4, 5, 6)))

    def test_ill_opts(self):
        with self.assertRaisesRegex(ValueError, ""):
            MosinLoss(sigmoid=True, softmax=True)
        chn_input = torch.ones((1, 1, 32, 32))
        chn_target = torch.ones((1, 1, 32, 32))
        with self.assertRaisesRegex(ValueError, ""):
            loss = MosinLoss(softmax=True)
            loss.forward(chn_input, chn_target)

    def test_input_warnings(self):
        chn_input = torch.ones((1, 1, 32, 32))
        chn_target = torch.ones((1, 1, 32, 32))
        with self.assertWarns(Warning):
            loss = MosinLoss(include_background=False)
            loss.forward(chn_input, chn_target)
        with self.assertWarns(Warning):
            loss = MosinLoss(use_base_loss=False, alpha=0.5)
        with self.assertWarns(Warning):
            loss = MosinLoss(use_base_loss=False, base_loss=DiceLoss())


if __name__ == "__main__":
    unittest.main()
