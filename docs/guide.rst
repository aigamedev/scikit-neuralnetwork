User Guide
==========

Regression
----------

Assuming your data is in the form of ``numpy.ndarray`` stored in the variables ``X_train`` and ``y_train`` you can train a :class:`sknn.mlp.Regressor` neural network.  The input and output arrays are continuous values in this case, but it's best if you normalize or standardize your inputs to the ``[0..1]`` or ``[-1..1]`` range. (See the :ref:`Pipeline` example below.)

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

It's also a good idea to normalize or standardize your data in this case too, for example using a :ref:`Pipeline` below.  The code here will train for 25 iterations.  Note that a ``Softmax`` output layer activation type is used here, and it's recommended as a default for classification problems.

.. code:: python

    y_example = nn.predict(X_example)

This code will run the classification with the neural network, and return a list of labels predicted for each of the example inputs.


Verbose Mode
------------

To see the output of the neural network's training, you need to configure two things: first setting up the Python logger (mandatory), and secondly to specify a verbose mode if you want more information during training (optional).

The first step is to configure either the ``sknn`` logger specifically, or do so globally (easier) as follows:

.. code:: python

    import sys
    import logging

    logging.basicConfig(
                format="%(message)s",
                level=logging.DEBUG,
                stream=sys.stdout)

Then you can optionally create your neural networks using an additional ``verbose`` parameter to show the output during training:

.. code:: python
    
    from sknn.mlp import Regressor, Layer

    nn = Regressor(
        layers=[Layer("Linear")],
        n_iter=20,
        verbose=1,
        valid_size=0.25)
    nn.fit(X, y)

This code will output a table containing validation scores at each of the twenty epochs.  The ``valid_size`` parameter is a ratio of the data to be used internally for validation; in short, the ``fit()`` function is automatically splitting the data into ``X_train`` and ``y_train`` as well as ``X_valid`` and ``y_valid``.


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


Pipeline
--------

Typically, neural networks perform better when their inputs have been normalized or standardized.  Using a scikit-learn's `pipeline <http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html>`_ support is an obvious choice to do this.

Here's how to setup such a pipeline with a multi-layer perceptron as a classifier:

.. code:: python

    from sknn.mlp improt Classifier, Layer

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import MinMaxScaler

    pipeline = Pipeline([
            ('min/max scaler', MinMaxScaler(feature_range=(0.0, 1.0))),
            ('neural network', Classifier(layers=[Layer("Softmax")], n_iter=25))])
    pipeline.fit(X_train, y_train)

You can thes use the pipeline as you would the neural network, or any other standard API from scikit-learn.
