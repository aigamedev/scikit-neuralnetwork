Misc. Additions
===============

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


Keyboard Interrupt
------------------

If you want to manually interrupt the main training loop by pressing ``CTRL+C`` but still finish the rest of your training script, you can wrap the call to fit with an exception handler:

.. code:: python

    # Setup experiment model and data.
    nn = mlp.Regressor(...)

    # Perform the gradient descent training.
    try:
        nn.fit(X, y)
    except KeyboardInterrupt:
        pass
    
    # Finalize the experiment here.
    print('score =', nn.score(X, y))

This was designed to work with both multi-layer perceptrons in :mod:`sknn.mlp` and auto-encoders in :mod:`sknn.ae`.  


CPU vs. GPU Platform
--------------------

To setup the library to use your GPU or CPU explicitly in 32-bit or 64-bit mode, you can use the ``platform`` pseudo-module.  It's a syntactic helper to setup the ``THEANO_FLAGS`` environment variable in a Pythonic way, for example:

.. code:: python

    # Use the GPU in 32-bit mode, falling back otherwise.
    from sknn.platform import gpu32
    
    # Use the CPU in 64-bit mode.
    from sknn.platform import cpu64

WARNING: This will only work if your program has not yet imported the ``theano`` module, due to the way that library is designed.  If ``THEANO_FLAGS`` are set on the command-line, they are not overwridden.


Backend Configuration
---------------------

As of version 0.3, ``scikit-neuralnetwork`` supports multiple neural network implementations called backends, each wrapped behind an identical standardized interface.  To configure a backend, you can do so by importing the corresponding module:

.. code:: python

    from sknn.backend import pylearn2

As long as you call this before creating a neural network, this will register the PyLearn2 implementation as the one that's used.  Supported backends are currently ``pylearn2`` (default) and ``deepy`` (incomplete).
