Input / Output
==============

Verbose Mode
------------

To see the output of the neural network's training, configure the Python logger called ``sknn`` or the default root logger.  This is possible using the standard ``logging`` module which you can setup as follows:

.. code:: python

    import sys
    import logging

    logging.basicConfig(
                format="%(message)s",
                level=logging.DEBUG,
                stream=sys.stdout)

Change the log level to ``logging.INFO`` for less information about each epoch, or ``logging.WARNING`` only to receive messages about problems or failures.

Using the flag ``verbose=True`` on either :class:`sknn.mlp.Classifier` and :class:`sknn.mlp.Regressor` will setup a default logger at ``DEBUG`` level if it does not exist, and ``verbose=False`` will setup a default logger at level ``WARNING`` if no logging has been configured.


Saving & Loading
----------------

To save a trained neural network to disk, you can do the following after having initialized your multi-layer perceptron as the variable ``nn`` and trained it:

.. code:: python

    import pickle
    pickle.dump(nn, open('nn.pkl', 'wb'))

After this, the file ``nn.pkl`` will be available in the current working directory — which you can reload at any time:

.. code:: python

    import pickle
    nn = pickle.load(open('nn.pkl', 'rb'))

In this case, you can use the reloaded multi-layer perceptron as if it had just been trained.  This will also work on different machines, whether CPU or GPU.

NOTE: You can serialize complex pipelines (for example from this section :ref:`example-pipeline`) using this exact same approach.


Extracting Parameters
---------------------

To access the weights and biases from the neural network layers, you can call the following function on any initialized neural network:

.. code:: python

    > nn.get_parameters()
    [Parameters(layer='hidden0', weights=array([[...]]), biases=array([[...]])),
     Parameters(layer='output', weights=array(...), biases=array(...))]

The list is ordered in the same way as the ``layers`` parameter passed to the constructor. Each item in the list is a named-tuple with ``names`` (string), ``weights`` and ``biases`` (both numpy.array).
