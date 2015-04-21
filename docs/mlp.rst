:mod:`sknn.mlp` —    Multi-Layer Perceptrons
=======================================

Most of the functionality provided by multi-layer perceptron is implemented in the (abstract) class :class:`sknn.mlp.BaseMLP`.  This documents all the construction parameters for Regressor and Classifier implementations, as well as helper functions.

.. autoclass:: sknn.mlp.BaseMLP


When using the multi-layer perceptron, you should initialize a Regressor or a Classifier directly.


Regression
----------

.. autoclass:: sknn.mlp.MultiLayerPerceptronRegressor
    :members:
    :inherited-members:


Classification
--------------

.. autoclass:: sknn.mlp.MultiLayerPerceptronClassifier
    :members:
    :inherited-members:
