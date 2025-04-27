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

The easiest way to install Topolosses (with all pre-built C++ extensions) is via PyPI:
Install Topolosses from `PyPI <https://pypi.org/project/topolosses/>`_:

.. code-block:: bash

   pip install topolosses

Alternative you can clone source-code from `github <https://github.com/J-falkenstein/topolosses>`_. 
**Note**: this requires manual building of the C++ extensions (see :ref:`working-with-source-code`):

Once installed, import and use any of the topology-aware losses just like a standard PyTorch loss:

.. code-block:: python

   from topolosses.losses import DiceLoss, BettiMatchingLoss

   # Combine topological (BettiMatchingLoss) with base component (DiceLoss)
   loss = BettiMatchingLoss(
      alpha=0.5,  # Weight for the topological component
      softmax=True,
      base_loss=DiceLoss(softmax=True, smooth=1e-3)
   )
   result = loss.forward(prediction, target)


Common Loss Structure
---------------------
Since most topology-aware loss functions combine the sparse topological component with a dense region loss like Dice to ensure both shape accuracy and topological correctness, this project follows the same approach. By default, it uses Dice as the base loss, but you can easily replace it with any custom loss you prefer—or even use just the topology component if that’s all you need.

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

.. _working-with-source-code:

Working with Source Code
-------------------------
If you want to modify the code (e.g., adjust a loss function), you’ll need to build the C++ extensions manually.
These extensions are only included in the PyPI wheels, not in the source code, so building them is required when working from source.

- build c++ extension explictly and include in your project
- build whole project yourself locally (python -m build) and then pip install {path}.whl

TODO explain how to install c++ extensions

Links
-----
- Pypi package: https://pypi.org/project/topolosses/
- Code: https://github.com/J-falkenstein/topolosses


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
