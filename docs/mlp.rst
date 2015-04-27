:mod:`sknn.mlp` — Multi-Layer Perceptrons
=========================================

Layer Specifications
--------------------

In this module, a neural network is made up of multiple layers — hence the name multi-layer perceptron!  You need to specify these layers by instantiating one of two types of specifications:

* :mod:`sknn.mlp.Layer`: A standard feed-forward layer that can use linear or non-linear activations.
* :mod:`sknn.mlp.Convolution`: An image-based convolve operation with shared weights, linear or not.

In practice, you need to create a list of these specifications and provide them as the ``layers`` parameter to the :class:`sknn.mlp.Regressor` or :class:`sknn.mlp.Classifier` constructors.

.. autoclass:: sknn.mlp.Layer

.. autoclass:: sknn.mlp.Convolution


MultiLayerPerceptron
--------------------

Most of the functionality provided to simulate and train multi-layer perceptron is implemented in the (abstract) class :class:`sknn.mlp.MultiLayerPerceptron`.  This class documents all the construction parameters for Regressor and Classifier derived classes (see below), as well as their various helper functions.

.. autoclass:: sknn.mlp.MultiLayerPerceptron

When using the multi-layer perceptron, you should initialize a Regressor or a Classifier directly.


Regressor
---------

See the class :class:`sknn.mlp.MultiLayerPerceptron` for inherited construction parameters.

.. autoclass:: sknn.mlp.Regressor
    :members:
    :inherited-members:


Classifier
----------

Also check the :class:`sknn.mlp.MultiLayerPerceptron` class for inherited construction parameters.

.. autoclass:: sknn.mlp.Classifier
    :members:
    :inherited-members:
