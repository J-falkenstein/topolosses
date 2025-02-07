# %%
import numpy as np
import matplotlib.pyplot as plt
import itertools
import sys
import torch
from matplotlib.colors import ListedColormap
from monai.losses import SoftDiceclDiceLoss, SoftclDiceLoss

# Add the parent directory to the system path
sys.path.append("..")


# %%
def create_paired_img_from_permutation(perm):
    image = torch.zeros((2, 2))
    image[0, 0] = perm[0]
    image[0, 1] = perm[1]
    image[1, 0] = perm[2]
    image[1, 1] = perm[3]
    return image


def create_imgs_from_permutation(perm1, perm2):
    image1 = create_paired_img_from_permutation(perm1)
    image2 = create_paired_img_from_permutation(perm2)
    paired_img = image1 + 2 * image2

    paired_img2 = paired_img.clone()
    paired_img2[paired_img2 == 0] = -5
    paired_img2[paired_img2 == 3] = 5
    return image1, image2, paired_img, paired_img2


cmap = ListedColormap(["white", "blue", "black", "purple"])

# %%
fig, ax = plt.subplots(4, 6, figsize=(20, 20))

for i, perm in enumerate(list(itertools.permutations([0, 1, 2, 3]))):
    image = create_paired_img_from_permutation(perm)

    j = i // 6
    k = i % 6

    ax[j, k].imshow(image, cmap=cmap)

# %%
preds = list(itertools.permutations([0, 0, 0, 0]))
gts = list(itertools.permutations([0, 1, 0, 1]))
print(preds)
pred_onehot, gt_onehot, draw_pairing, paired_img = create_imgs_from_permutation(preds[0], gts[1])

fig, ax = plt.subplots(1, 3, figsize=(20, 20))
ax[0].imshow(pred_onehot, cmap=cmap)
ax[1].imshow(gt_onehot, cmap=cmap)
ax[2].imshow(draw_pairing, cmap=cmap)

# %%
import importlib
import app.losses.cldice.src.cldice_loss as cldice_loss
import tests.losses_original.dice_losses as cldice_loss_orignal

importlib.reload(cldice_loss)
importlib.reload(cldice_loss_orignal)


def compare_loss_implementation(pred, gt, expected_result, ignore_background=True, num_classes=2):
    # TODO expected result does not make sense anymore, but need
    # fig, ax = plt.subplots(1, 2, figsize=(20, 20))
    # ax[0].imshow(pred[0], cmap=cmap)
    # ax[1].imshow(gt[0], cmap=cmap)
    # plt.show()

    pred_onehot = torch.nn.functional.one_hot(pred.to(torch.int64), num_classes).permute(0, 3, 1, 2).float()
    gt_onehot = torch.nn.functional.one_hot(gt.to(torch.int64), num_classes).permute(0, 3, 1, 2).float()

    cldice_new = cldice_loss.DiceCLDiceLoss(weights=[0.0, 1.0])
    cldice_orignal = cldice_loss_orignal.Multiclass_CLDice()
    dicecldice_monai = SoftDiceclDiceLoss(smooth=1e-5)
    cldice_monai = SoftclDiceLoss(smooth=1e-5)

    loss = cldice_new(pred_onehot, gt_onehot)
    loss_original = cldice_orignal(pred_onehot, gt_onehot)
    loss_monai_dicecldice = dicecldice_monai(pred_onehot, gt_onehot)
    loss_monai_cldice = cldice_monai(pred_onehot, gt_onehot)

    print("Refactored implementation: ", loss)
    print("Original implementation: ", loss_original)
    print("Monai dice cl dice: ", loss_monai_dicecldice)
    print("Monai cl dice: ", loss_monai_cldice)


pred = torch.tensor(
    [
        [
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    ]
).float()

gt = torch.tensor(
    [
        [
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    ]
).float()

compare_loss_implementation(pred, gt, 3, ignore_background=True)

pred = torch.tensor(
    [
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    ]
).float()

gt = torch.tensor(
    [
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    ]
).float()

compare_loss_implementation(pred, gt, 0, ignore_background=True)

gt = torch.tensor(
    [
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    ]
).float()

pred = torch.tensor(
    [
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    ]
).float()

compare_loss_implementation(pred, gt, 0, ignore_background=True)

gt = torch.tensor(
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
).float()

pred = torch.tensor(
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
).float()

compare_loss_implementation(pred, gt, 1, ignore_background=True)

gt = (
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
    ).float()
)

pred = (
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
    ).float()
)

compare_loss_implementation(pred, gt, 1, ignore_background=True)

pred = torch.tensor(
    [
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    ]
).float()

gt = torch.tensor(
    [
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    ]
).float()

compare_loss_implementation(pred, gt, 0, ignore_background=True)

pred = torch.tensor(
    [
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
        ]
    ]
).float()

gt = torch.tensor(
    [
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    ]
).float()

compare_loss_implementation(pred, gt, 0, ignore_background=True)

pred = torch.tensor(
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
).float()

gt = torch.tensor(
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
).float()

compare_loss_implementation(pred, gt, 0, ignore_background=True)

pred = torch.tensor(
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
).float()

gt = torch.tensor(
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
).float()

compare_loss_implementation(pred, gt, 1, ignore_background=True)

# %%
