flow_segmenter
==============

Python package for document image segmentation using YOLO and Kraken models,
developed for the `Flow Project <https://flow-project.net>`_.

.. toctree::
   :maxdepth: 3
   :caption: Contents:

API Reference
=============
Main Package
------------
.. autosummary::
   :toctree: _autosummary
   :recursive:

   flow_segmenter

Core Modules
------------
Segmenter
~~~~~~~~~
.. autosummary::
   :toctree: _autosummary

   flow_segmenter.segmenter.SegmenterYolo
   flow_segmenter.segmenter.SegmenterKrakenLinemasks

Configuration
~~~~~~~~~~~~~
.. autosummary::
   :toctree: _autosummary

   flow_segmenter.config.SegmenterConfig
   flow_segmenter.config.SegmenterBaseConfig

Utilities
~~~~~~~~~
.. autosummary::
   :toctree: _autosummary

   flow_segmenter.xml_utils.XMLUtils
   flow_segmenter.baseline_utils.BaselineUtils

Exceptions
~~~~~~~~~~
.. autosummary::
   :toctree: _autosummary

   flow_segmenter.exceptions

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
