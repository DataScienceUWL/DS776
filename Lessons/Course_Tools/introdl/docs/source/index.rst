DS776 introdl Package Documentation
=====================================

Welcome to the DS776 Deep Learning Course Package documentation.

The ``introdl`` package provides utilities, training functions, and visualization tools
for the DS776 Deep Learning course. This documentation includes comprehensive examples
from course lessons L01-L06.

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   api/core_utilities
   api/training_functions
   api/visualization_functions
   api/storage_utilities

.. toctree::
   :maxdepth: 1
   :caption: Usage Examples:

   examples/lesson_examples
   examples/common_patterns

Quick Start
-----------

.. code-block:: python

   from introdl.utils import config_paths_keys, get_device
   from introdl.idlmam import train_network
   from introdl.visul import plot_training_metrics

   # Setup environment
   paths = config_paths_keys()
   device = get_device()

   # Train and visualize
   results = train_network(model, loss_func, train_loader, device=device)
   plot_training_metrics(results, [['train loss', 'test loss']])

Package Overview
---------------

**Core Modules:**

* :mod:`introdl.utils` - Core utilities and configuration
* :mod:`introdl.idlmam` - Training functions and algorithms
* :mod:`introdl.visul` - Visualization and plotting functions
* :mod:`introdl.nlp` - Natural language processing utilities
* :mod:`introdl.generation` - Text generation functions
* :mod:`introdl.summarization` - Text summarization functions

**Private Modules:**

* :mod:`introdl._storage` - Storage management utilities
* :mod:`introdl._paths` - Path resolution utilities

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
