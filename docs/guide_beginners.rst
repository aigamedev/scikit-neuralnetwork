Simple Examples
===============

.. _example-regression:

Regression
----------

Assuming your data is in the form of ``numpy.ndarray`` stored in the variables ``X_train`` and ``y_train`` you can train a :class:`sknn.mlp.Regressor` neural network.  The input and output arrays are continuous values in this case, but it's best if you normalize or standardize your inputs to the ``[0..1]`` or ``[-1..1]`` range. (See the :ref:`example-pipeline` example below.)

.. code:: python

    from sknn.mlp import Regressor, Layer

    nn = Regressor(
        layers=[
            Layer("Rectifier", units=100),
            Layer("Linear")],
        learning_rate=0.02,
        n_iter=10)
    nn.fit(X_train, y_train)

This will train the regressor for 10 epochs (specified via the ``n_iter`` parameter).  The ``layers`` parameter specifies how the neural network is structured; see the :class:`sknn.mlp.Layer` documentation for supported layer types and parameters.

Then you can use the trained NN as follows:

.. code:: python

    y_example = nn.predict(X_example)

This will return a new ``numpy.ndarray`` with the results of the feed-forward simulation of the network and the estimates given the input features.


.. _example-classification:

Classification
--------------

If your data in ``numpy.ndarray`` contains integer labels as outputs and you want to train a neural network to classify the data, use the following snippet:

.. code:: python

    from sknn.mlp import Classifier, Layer

    nn = Classifier(
        layers=[
            Layer("Maxout", units=100, pieces=2),
            Layer("Softmax")],
        learning_rate=0.001,
        n_iter=25)
    nn.fit(X_train, y_train)

It's also a good idea to normalize or standardize your data in this case too, for example using a :ref:`example-pipeline` below.  The code here will train for 25 iterations.  Note that a ``Softmax`` output layer activation type is used here, and it's recommended as a default for classification problems.

If you want to do multi-label classification, simply fit using a ``y`` array of integers that has multiple dimensions, e.g. shape `(N, 3)` for three different classes.  Then, make sure the last layer is ``Sigmoid`` instead.

.. code:: python

    y_example = nn.predict(X_example)

This code will run the classification with the neural network, and return a list of labels predicted for each of the example inputs.  If you need to access the probabilities for the predictions, use ``predict_proba()`` and see the content of the ``classes_`` property that provides the labels for each features, which you can use to compute the probability indices.


.. _example-convolution:

Convolution
-----------

Working with images as inputs in 2D (as greyscale) or 3D (as RGB) images stored in ``numpy.ndarray``, you can use convolution to train a neural network with shared weights.  Here's an example how classification would work:

.. code:: python

    from sknn.mlp import Classifier, Convolution, Layer

    nn = Classifier(
        layers=[
            Convolution("Rectifier", channels=8, kernel_shape=(3,3)),
            Layer("Softmax")],
        learning_rate=0.02,
        n_iter=5)
    nn.fit(X_train, y_train)

The neural network here is trained with eight kernels of shared weights in a ``3x3`` matrix, each outputting to its own channel.  The rest of the code remains the same, but see the :class:`sknn.mlp.Layer` documentation for supported convolution layer types and parameters.


Per-Sample Weighting
--------------------

When training a classifier with data that has unbalanced labels, it's useful to adjust the weight of the different training samples to prevent bias.  This is achieved via a feature called masking.  You can specify the weights of each training sample when calling the ``fit()`` function.

.. code:: python

    w_train = numpy.array((X_train.shape[0],))
    w_train[y_train == 0] = 1.2
    w_train[y_train == 1] = 0.8

    nn.fit(X_train, y_train, w_train)

In this case, there are two classes ``0`` given weight ``1.2``, and ``1`` with weighting ``0.8``.  This feature also works for regressors as well.