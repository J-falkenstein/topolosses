# topolosses

This is the readme used for the github repo. Hence, the entire project including the setup and test files. 

# topolosses

Topolosses is a Python package providing topology-aware losses for segmentation tasks. It includes losses that improve topological properties in segmentation models, such as `DiceLoss`, `TopographLoss`, and `BettiMatchingLoss`.

## Installation

Install the package from Test PyPI. 
The verision 0.1.2 includes betti matching:

```bash
pip install -i https://test.pypi.org/simple/ topolosses==0.1.2
```

## Usage

Import the desired loss functions:

```python
from topolosses.losses import CLDiceLoss, DiceLoss, BettiMatchingLoss

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
```

## Common Arguments for Loss Functions

- **`include_background`** (bool):  
  Includes the background in the topology-aware component computation. Default: `False`.

- **`alpha`** (float):  
  Weight for combining the topology-aware component and the base loss component. Default: `0.5`.

- **`sigmoid`** (bool):  
  Applies sigmoid activation before computing the topology-aware component. Default: `False`.

- **`softmax`** (bool):  
  Applies softmax activation before computing the topology-aware component. Default: `False`.

- **`use_base_component`** (bool):  
  If `False`, only the topology-aware component is computed. Default: `True`.

- **`base_loss`** (_Loss, optional):  
  The base loss function used with the topology-aware component. Default: `None`.

> **Note**: Each loss function also has specific arguments that are unique to its behavior. These are documented within the code using docstrings, and can be easily accessed using Python's `help()` function or by exploring the source code.


## Folder Structure


```
topolosses
├─ .DS_Store
├─ CMakeLists.txt
├─ LICENSE
├─ README.md
├─ pyproject.toml
└─ topolosses
   ├─ README.md
   ├─ __init__.py
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