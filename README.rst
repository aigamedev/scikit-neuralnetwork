scikit-neuralnetwork
====================

Deep neural network implementation without the learning cliff!  This library implements multi-layer perceptrons as a wrapper for the powerful ``pylearn2`` library that's compatible with ``scikit-learn`` for a more user-friendly and Pythonic interface. Oh, and it runs on your GPU by default.

**NOTE**: This project is possible thanks to the `nucl.ai Conference <http://nucl.ai/>`_ on **July 20-22**. Join us in **Vienna**!

|Build Status| |Documentation Status| |Code Coverage|

----

Installation
------------

You'll need to first install some dependencies manually.  Unfortunately, ``pylearn2`` isn't yet installable via PyPI and recommends an editable (``pip -e``) installation::

    > git clone https://github.com/lisa-lab/pylearn2.git
    > cd pylearn2; python setup.py develop

Once that's done, you can grab this repository and set your ``PYTHONPATH`` to point to the correct folder, or install from ``setup.py`` in the exact same way.


Demonstration
-------------

To run a visualization that uses the `sknn.mlp.MultiLayerPerceptron` just run the following command::

    > PYTHONPATH=. python examples/plot_mlp.py --params activation

The datasets are randomized each time, but the output should be an image that looks like this...

.. image:: docs/plot_activation.png


Benchmarks
----------

Here are the results of testing 10 epochs of training for two-thirds of the original MNIST data, on Ubuntu 14.04 and a GeForce GTX 650 (Memory: 1024Mb, Cores: 384).  You can run ``examples/bench_mnist.py`` to get the results.

.. class:: center

==========  ============  ===============  ===================
   MNIST      sknn.mlp      nolearn.dbn      nolearn.lasagne
==========  ============  ===============  ===================
 Accuracy    **98.00%**       97.80%             97.75%
 Training     **36s**          274s                68s
==========  ============  ===============  ===================

All the networks have 300 hidden units of the default type, and were given the same data with monitoring disabled. (For ``sknn`` the monitoring was commented out manually as of 2015/04/10.)  The remaining third of the MNIST dataset was only used to test the score once training terminated.

**WARNING**: These results are not surprising, as ``pylearn2`` is developed by one of the best and most famous Deep Learning labs in the world.  However, they are not definitive and those numbers are very sensitive to parameter changes.


Getting Started
---------------

The library supports both regressors (to estimate continuous outputs) and classifiers (to predict classes).  This is ``sklearn`` compatible API::

    import sknn.mlp

    nn = sknn.mlp.MultiLayerPerceptronClassifier(
        layers=[("Rectifier", 100), ("Linear",)],
        learning_rate=0.02,
        n_iter=10)

    nn.fit(X_train, y_train)
    nn.predict(X_test)

    nn.score(X_valid, y_valid)

We currently recommend reading ``mlp.py`` for more information about the parameters.  There's also `generated documentation <http://scikit-neuralnetwork.readthedocs.org/>`_ for details on the construction paramaters.


.. |Build Status| image:: https://travis-ci.org/aigamedev/scikit-neuralnetwork.svg?branch=master
   :target: https://travis-ci.org/aigamedev/scikit-neuralnetwork

.. |Documentation Status| image:: https://readthedocs.org/projects/scikit-neuralnetwork/badge/?version=latest
    :target: http://scikit-neuralnetwork.readthedocs.org/

.. |Code Coverage| image:: https://coveralls.io/repos/aigamedev/scikit-neuralnetwork/badge.svg?branch=master
    :target: https://coveralls.io/r/aigamedev/scikit-neuralnetwork?branch=master
