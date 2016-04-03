Low-Level Configuration
=======================

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

WARNING: This will only work if your program has not yet imported the ``theano`` module, due to the way that library is designed.  If ``THEANO_FLAGS`` are set on the command-line, they are not overridden.


Multiple Threads
----------------

In CPU mode and on supported platforms (e.g. gcc on Linux), to use multiple threads (by default the number of processors) you can also import from the ``platform`` pseudo-module as follows:

.. code:: python

    # Use the maximum number of threads for this script.
    from sknn.platform import cpu32, threading

If you want to specify the number of threads exactly, you can import for example ``threads2`` or ``threads8`` — or any other positive number that's supported by your OS.  Alternatively, you can manually set these values by using the ``OMP_NUM_THREADS`` environment variable directly, and setting ``THEANO_FLAGS`` to include ``openmp=True``.


Backend Configuration
---------------------

As of version 0.3, ``scikit-neuralnetwork`` supports multiple neural network implementations called backends, each wrapped behind an identical standardized interface.  To configure a backend, you can do so by importing the corresponding module:

.. code:: python

    from sknn.backend import lasagne

As long as you call this before creating a neural network, this will register the PyLearn2 implementation as the one that's used.  Supported backends are currently ``lasagne`` (default) and ``pylearn2`` (removed).
