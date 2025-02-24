# topolosses

This is the readme used for the github repo. Hence, the entire project including the setup and test files. 
```
topolosses
├─ .DS_Store
├─ CMakeLists.txt
├─ LICENSE
├─ README.md
├─ pyproject.toml
├─ tests
│  ├─ cldice_tests.py
│  ├─ data_examples
│  │  ├─ CREMI
│  │  │  ├─ .DS_Store
│  │  │  ├─ images
│  │  │  │  ├─ image_0114.png
│  │  │  │  ├─ image_0117.png
│  │  │  │  └─ image_0123.png
│  │  │  └─ labels
│  │  │     ├─ label_0114.png
│  │  │     ├─ label_0117.png
│  │  │     └─ label_0123.png
│  │  ├─ buildings
│  │  │  ├─ .DS_Store
│  │  │  ├─ images
│  │  │  │  ├─ image_0090.png
│  │  │  │  └─ image_0105.png
│  │  │  └─ labels
│  │  │     ├─ label_0090.png
│  │  │     └─ label_0105.png
│  │  ├─ colon_cancer_cells
│  │  │  ├─ .DS_Store
│  │  │  ├─ images
│  │  │  │  ├─ image_0020.png
│  │  │  │  └─ image_0022.png
│  │  │  └─ labels
│  │  │     ├─ label_0020.png
│  │  │     └─ label_0022.png
│  │  ├─ elegans_cells
│  │  │  ├─ .DS_Store
│  │  │  ├─ images
│  │  │  │  ├─ image_0090.png
│  │  │  │  └─ image_0095.png
│  │  │  └─ labels
│  │  │     ├─ label_0090.png
│  │  │     └─ label_0095.png
│  │  └─ roads
│  │     ├─ .DS_Store
│  │     ├─ images
│  │     │  ├─ image_0111.png
│  │     │  ├─ image_0113.png
│  │     │  └─ image_0122.png
│  │     └─ labels
│  │        ├─ label_0111.png
│  │        ├─ label_0113.png
│  │        └─ label_0122.png
│  ├─ losses_original
│  │  ├─ betti_losses.py
│  │  ├─ dice_losses.py
│  │  ├─ exact_topograph.py
│  │  ├─ hutopo.py
│  │  ├─ mosin.py
│  │  ├─ topograph.py
│  │  └─ utils.py
│  ├─ test_cldice_loss.py
│  ├─ test_dice_loss.py
│  └─ test_utils.py
└─ topolosses
   ├─ README.md
   ├─ __init__.py
   └─ losses
      ├─ __init__.py
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
      │     │  ├─ topograph.cpp
      │     │  └─ topograph.hpp
      │     ├─ setup.py
      │     └─ topograph_loss.py
      └─ utils.py

```