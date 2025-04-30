<!-- This is the readme used for the github repo and later not the one for pypi. Hence, the entire project including the setup and test files.  
currently it is the only readme and used also for pypi -->

# topolosses

Topolosses is a Python package providing topology-aware losses for segmentation tasks. It includes losses that improve topological properties in segmentation models, such as `DiceLoss`, `TopographLoss`, `BettiMatchingLoss`, `HutopoLoss`, `MosinLoss` and `WarpingLoss`.

The documentation is available here: https://j-falkenstein.github.io/topolosses/

## Installation

You can install topolosses directly from PyPI:

```bash
pip install topolosses
```

Pre-built wheels are available for Linux platforms.
If no compatible wheel exists, pip will compile from source locally. See working-with-source-code section for more details. 

## Usage

Import the desired loss functions and implement the loss functions like any standard PyTorch loss:

```python
from topolosses.losses import CLDiceLoss, DiceLoss, BettiMatchingLoss

# Create a c CLDice loss (which itself combines with Dice)
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

# Combine topological (BettiMatchingLoss) with base component (CLDiceLoss)
loss = BettiMatchingLoss(
    alpha=0.5,  # Weight for the topological component
    softmax=True,
    base_loss=clDiceLoss
)

result = loss.forward(prediction, target)
```

## Common Arguments for Loss Functions
Since most topology-aware loss functions combine the sparse topological component with a dense region loss like Dice to ensure both shape accuracy and topological correctness, this project follows the same approach. By default, it uses Dice as the base loss, but you can easily replace it with any custom loss you prefer—or even use just the topology component if that’s all you need.

- **`alpha`** (float):  
  Weight for combining the topology-aware component and the base loss component. Default: `0.5`.

- **`sigmoid`** (bool):  
  Applies sigmoid activation to the forward pass input before computing the topology-aware component. 
  If using the default Dice loss, the sigmoid-transformed input is also used. For custom base losses, the raw input is passed. Default: `False`.

- **`softmax`** (bool):  
  Applies softmax activation to the forward pass input before computing the topology-aware component. 
  If using the default Dice loss, the softmax-transformed input is also used. For custom base losses, the raw input is passed. Default: `False`.

- **`use_base_component`** (bool):  
  If `False`, only the topology-aware component is computed. Default: `True`.

- **`base_loss`** (_Loss, optional):  
  The base loss function used with the topology-aware component. Default: `None`.

> **Note**: Each loss function also has specific arguments. These are documented within the code using docstrings, and can be easily accessed using Python's `help()` function or by exploring the source code. The API reference is available here: https://j-falkenstein.github.io/topolosses/


## Working with Source Code
If no binary for your plattform is available or if you want to modify the code (e.g., adjust a loss function), you’ll need to build the C++ extensions locally. 

If no compatible wheel exists, pip will compile from source locally. 
To compile the C++ extension you require a C++ compiler, Python development headers, OpenCV, Boost, and Eigen libraries. (TODO: specify which versions are needed and where to locate the libraries).
However, because this approach is very error prone it is better to clone source-code from `GitHub <https://github.com/J-falkenstein/topolosses>`_. 
You can tweak pyproject.toml and CMakeLists.txt to point at your local library paths.

- Option 1: After cloning the repo you can tweak pyproject.toml and CMakeLists.txt to point at your local library paths. Then you can use python -m build to build the wheels and pip install {path}.whl.
- Option 2: When not wanting to build but working directly inside the package it requires manual building of the C++ extensions. Might require adjusting the import statements. (TODO explain how to install c++ extensions)



## Folder Structure


```
topolosses
├─ CMakeLists.txt
├─ LICENSE
├─ README.md
├─ pyproject.toml
├─ docs
   ├─ index.rst
   ├─ ...
└─ topolosses
   ├─ README.md
   ├─ __init__.py
   ├─ metrics
   └─ losses
      ├─ __init__.py
      ├─ betti_matching
      │  ├─ __init__.py
      │  └─ src
      │     ├─ betti_matching_loss.py
      │     └─ ext
      │        └─ Betti-Matching-3D
      │           ├─ CMakeLists.txt
      │           ├─ LICENSE
      │           ├─ README.md
      │           ├─ src
      │           │  ├─ BettiMatching.cpp
      │           │  ├─ BettiMatching.h
      │           │  ├─ _BettiMatching.cpp
      │           │  ├─ config.h
      │           │  ├─ data_structures.cpp
      │           │  ├─ data_structures.h
      │           │  ├─ main.cpp
      │           │  ├─ npy.hpp
      │           │  ├─ src_1D
      │           │  │  ├─ 
      │           │  ├─ src_2D
      │           │  │  ├─ 
      │           │  ├─ src_3D
      │           │  │  ├─ 
      │           │  ├─ src_nD
      │           │  │  ├─ 
      │           │  ├─ utils.cpp
      │           │  └─ utils.h
      │           └─ utils
      │              ├─ functions.py
      │              └─ plots.py
      ├─ cldice
      │  ├─ __init__.py
      │  └─ src
      │     └─ cldice_loss.py
      ├─ dice
      │  ├─ __init__.py
      │  └─ src
      │     └─ dice_loss.py
      ├─ topograph
      │  ├─ __init__.py
      │  └─ src
      │     ├─ ext
      │     │  ├─ _topograph.cpp
      │     │  ├─ setup.py
      │     │  ├─ topograph.cpp
      │     │  └─ topograph.hpp
      │     └─ topograph_loss.py
      └─ utils.py

```