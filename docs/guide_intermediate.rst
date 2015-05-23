Misc. Additions
===============

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
        verbose=True,
        valid_size=0.25)
    nn.fit(X, y)

This code will output a table containing validation scores at each of the twenty epochs.  The ``valid_size`` parameter is a ratio of the data to be used internally for validation; in short, the ``fit()`` function is automatically splitting the data into ``X_train`` and ``y_train`` as well as ``X_valid`` and ``y_valid``.


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
