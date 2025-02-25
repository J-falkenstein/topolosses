#%% 
import numpy as np
import matplotlib.pyplot as plt
import itertools
import sys
import torch
from matplotlib.colors import ListedColormap

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
    paired_img2[paired_img2==0] = -5
    paired_img2[paired_img2==3] = 5
    return image1, image2, paired_img, paired_img2

cmap = ListedColormap(['white', 'blue', 'black', 'purple'])

# %%
fig, ax = plt.subplots(4, 6, figsize=(20, 20))

for i, perm in enumerate(list(itertools.permutations([0, 1, 2, 3]))):
    image = create_paired_img_from_permutation(perm)

    j = i // 6
    k = i % 6

    ax[j, k].imshow(image, cmap=cmap)

#%%
preds = list(itertools.permutations([0,0,0,0]))
gts = list(itertools.permutations([0,1,0,1]))
print(preds)
pred_onehot, gt_onehot, draw_pairing, paired_img = create_imgs_from_permutation(preds[0], gts[1])

fig, ax = plt.subplots(1, 3, figsize=(20, 20))
ax[0].imshow(pred_onehot,cmap=cmap)
ax[1].imshow(gt_onehot, cmap=cmap)
ax[2].imshow(draw_pairing, cmap=cmap)

# %%
import importlib
import metrics.topograph as topograph_metric
importlib.reload(topograph_metric)


def test_metric(module, pred, gt, expected_result, ignore_background=True, num_classes=2):
    fig, ax = plt.subplots(1, 2, figsize=(20, 20))
    ax[0].imshow(pred[0], cmap=cmap)
    ax[1].imshow(gt[0], cmap=cmap)
    plt.show()
    
    pred_onehot = torch.nn.functional.one_hot(pred.to(torch.int64), num_classes).permute(0, 3, 1, 2).float()
    gt_onehot = torch.nn.functional.one_hot(gt.to(torch.int64), num_classes).permute(0, 3, 1, 2).float()


    topograph_metric_8 = module.TopographMetric(num_processes=1, ignore_background=ignore_background, eight_connectivity=True)

    metric_res_8 = topograph_metric_8(pred_onehot, gt_onehot)

    print("Metric: ", metric_res_8, " Expected: ", expected_result)
    if metric_res_8 != expected_result:
        raise ValueError("Test failed")


pred = torch.tensor([[[1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 0.0, 0.0, 0.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0]]]).float()

gt = torch.tensor([[[0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0]]]).float()

test_metric(topograph_metric, pred, gt, 3, ignore_background=True)

pred = torch.tensor([[[0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0]]]).float()

gt = torch.tensor([[[0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 1.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0, 1.0, 0.0],
                    [0.0, 1.0, 1.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0]]]).float()

test_metric(topograph_metric, pred, gt, 0, ignore_background=True)

gt = torch.tensor([[[0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0]]]).float()

pred = torch.tensor([[[0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 1.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0, 1.0, 0.0],
                    [0.0, 1.0, 1.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0]]]).float()

test_metric(topograph_metric, pred, gt, 0, ignore_background=True)

gt = torch.tensor([[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]]).float()

pred = torch.tensor([[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]]).float()

test_metric(topograph_metric, pred, gt, 1, ignore_background=True)

gt = 1 - torch.tensor([[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]]).float()

pred = 1 - torch.tensor([[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]]).float()

test_metric(topograph_metric, pred, gt, 1, ignore_background=True)

pred = torch.tensor([[[0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0]]]).float()

gt = torch.tensor([[[0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0]]]).float()

test_metric(topograph_metric, pred, gt, 0, ignore_background=True)

pred = torch.tensor([[[0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 1.0, 1.0, 0.0],
                    [0.0, 1.0, 1.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0]]]).float()

gt = torch.tensor([[[0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 1.0, 0.0, 0.0],
                    [0.0, 1.0, 1.0, 0.0, 0.0],
                    [0.0, 1.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0]]]).float()

test_metric(topograph_metric, pred, gt, 0, ignore_background=True)

pred = torch.tensor([[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                    [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]]).float()

gt = torch.tensor([[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                    [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]]).float()

test_metric(topograph_metric, pred, gt, 0, ignore_background=True)

pred = torch.tensor([[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                    [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]]).float()

gt = torch.tensor([[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                    [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]]).float()

test_metric(topograph_metric, pred, gt, 1, ignore_background=True)
# %%
import importlib
import losses.exact_topograph as exact_topograph
importlib.reload(exact_topograph)
pred_np = pred_onehot[0,1].numpy()
gt_np = gt_onehot[0,1].numpy()

print(pred_onehot.shape)
import losses.utils as utils
import timeit

# create random 200x200 image
pred_np = torch.rand((1,2,1000, 1000), device="cuda").float()
gt_np = torch.rand((1,2,1000, 1000), device="cuda").float()

print(pred_np)

exact_topograph_criteria = exact_topograph.ExactTopographLoss(
    include_background=True,
    num_processes=1
)

new_topograph_criteria = topograph.TopographLoss(
    include_background=True, 
    num_processes=1
)


#test.single_sample_loss((pred_np, gt_np, 1))
#new_time = timeit.timeit("new_topograph_criteria(pred_np, gt_np)", globals=globals(), number=1)
#print(new_time * 1000)
time_it = timeit.timeit("exact_topograph_criteria(pred_np, gt_np)", globals=globals(), number=1)
print(time_it * 1000)

# %%
import torch.nn.functional as F
import torch
import sys
# Add the parent directory to the system path
sys.path.append("..")
from losses.utils import aufdickung

filter = torch.ones((8,8))

filter_1 = F.pad(filter, (1,1,1,1), value=0)
filter_2 = F.pad(filter, (1,1,1,1), value=1)
print(filter_1)
print(filter_2)

input = torch.tensor([[1,0,1], [0,0,0], [1,1, 0]])

smooth_img = F.conv_transpose2d(input.unsqueeze(0).unsqueeze(0).float(), filter_1.unsqueeze(0).unsqueeze(0), stride=5, padding=2)
smooth_img[smooth_img > 0] = 1
print(input.numpy())
print(smooth_img[0,0].numpy())
print(smooth_img.shape)

print(aufdickung(input.numpy()).shape)
print(aufdickung(input.numpy(), 2))
# %%
import sys
# Add the parent directory to the system path
sys.path.append("..")
from losses import utils

prem = [0, 1, 1, 2]
utils.create_permuation_plot(prem)
# %%
