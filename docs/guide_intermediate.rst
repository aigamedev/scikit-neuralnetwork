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

    import pickle
    nn == pickle.load(open('nn.pkl', 'rb'))

In this case, you can use the reloaded multi-layer perceptron as if it had just been trained.  This will also work on different machines, whether CPU or GPU.

NOTE: You can serialize complex pipelines (for example from this section :ref:`example-pipeline`) using this exact same approach.


GPU Backend
-----------

To setup the library to use your GPU or CPU explicitly in 32-bit or 64-bit mode, you can use the ``backend`` pseudo-module.  It's a syntactic helper to setup ``THEANO_FLAGS`` in a Pythonic way, for example:

.. code:: python

    # Use the GPU in 32-bit mode, falling back otherwise.
    from sknn.backend import gpu32
    
    # Use the CPU in 64-bit mode.
    from sknn.backend import cpu64

WARNING: This will only work if your program has not yet imported the ``theano`` module, due to the way the library is designed.  If ``THEANO_FLAGS`` are set on the command-line, they are not overwridden.
