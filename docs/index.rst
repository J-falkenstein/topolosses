.. topolosses documentation master file, created by
   sphinx-quickstart on Wed Apr 23 16:45:41 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


Topolosses Documentation
========================

Topolosses is a Python package providing topology-aware loss functions for segmentation tasks.  
It contains losses designed to improve the topological correctness of model predictions, such as CLDiceLoss, BettiMatchingLoss, TopographLoss, and more.


Getting Started
---------------

Install Topolosses from PyPI:

.. code-block:: bash
   pip install topolosses

.. code-block:: python
   from topolosses.losses import DiceLoss, BettiMatchingLoss
   # ... create and combine losses as needed

Common Loss Structure
---------------------
Since most Topologliy aware loss function combine the topolgoy componebent with a component like dice, the Loss function in this project do this all. 
By default this base combonent is a simple dice loss but the use can adjust this to fully custom loss function. 

- **alpha** (*float*):  
  Weight for combining the topology-aware component and the base loss component. Default: ``0.5``.

- **sigmoid** (*bool*):  
  Applies sigmoid activation to the forward-pass input before computing the topology-aware component.  
  If using the default Dice loss, the sigmoid-transformed input is also used; for a custom base loss, the raw input is passed. Default: ``False``.

- **softmax** (*bool*):  
  Applies softmax activation to the forward-pass input before computing the topology-aware component.  
  If using the default Dice loss, the softmax-transformed input is also used; for a custom base loss, the raw input is passed. Default: ``False``.

- **use_base_component** (*bool*):  
  If ``False``, only the topology-aware component is computed. Default: ``True``.

- **base_loss** (*Loss*, optional):  
  The base loss function used with the topology-aware component. Default: ``None``.


API References
---------------

.. toctree::
   :maxdepth: 1
   :caption: Losses:

   topolosses.losses.betti_matching
   topolosses.losses.cldice
   topolosses.losses.dice
   topolosses.losses.hutopo
   topolosses.losses.mosin
   topolosses.losses.topograph
   topolosses.losses.warping

.. toctree::
   :maxdepth: 1
   :caption: utils:

   topolosses.losses.utils

Working with Source Code
---------------
If you want to modify the code (e.g., adjust a loss function), you’ll need to build the C++ extensions manually.
These extensions are only included in the PyPI wheels, not in the source code, so building them is required when working from source.

TODO explain how to install c++ extensions 

Indices and tables
---------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
