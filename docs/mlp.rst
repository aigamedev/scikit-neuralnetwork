:mod:`sknn.mlp` —    Multi-Layer Perceptrons
=======================================

Most of the functionality provided to simulate and train multi-layer perceptron is implemented in the (abstract) class :class:`sknn.mlp.BaseMLP`.  This class documents all the construction parameters for Regressor and Classifier derived classes (see below), as well as their various helper functions.

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
