:mod:`sknn.ae` — Auto-Encoders
==============================

In this module, a neural network is made up of stacked layers of weights that encode input data (upwards pass) and then decode it again (downward pass).  This is implemented in layers:

* :mod:`sknn.ae.Layer`: Used to specify an upward and downward layer with non-linear activations.

In practice, you need to create a list of these specifications and provide them as the ``layers`` parameter to the :class:`sknn.ae.AutoEncoder` constructor.


Layer Specifications
--------------------

.. autoclass:: sknn.ae.Layer


Auto-Encoding Transformers
--------------------------

This class serves two high-level purposes:

    1. **Unsupervised Learning —** Provide a form of unsupervised learning to train weights in each layer.  These weights can then be reused in a :class:`sknn.mlp.MultiLayerPerceptron` for better pre-training.
    2. **Pipeline Transformation —** Encode inputs into an intermediate representation for use in a pipeline, for example to reduce the dimensionality of an input vector using stochastic gradient descent.

.. autoclass:: sknn.ae.AutoEncoder
