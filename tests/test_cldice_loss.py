## TODO: adjust entire testing strategy make it more concise and easier to understand


import numpy as np
import matplotlib.pyplot as plt
import itertools
import sys
import torch
from matplotlib.colors import ListedColormap
from monai.losses import SoftDiceclDiceLoss, SoftclDiceLoss, DiceLoss


# Add the parent directory to the system path
sys.path.append("..")


# Test with wro
