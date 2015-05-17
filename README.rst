scikit-neuralnetwork
====================

Deep neural network implementation without the learning cliff!  This library implements multi-layer perceptrons as a wrapper for the powerful ``pylearn2`` library that's compatible with ``scikit-learn`` for a more user-friendly and Pythonic interface.

**NOTE**: This project is possible thanks to the `nucl.ai Conference <http://nucl.ai/>`_ on **July 20-22**. Join us in **Vienna**!

|Build Status| |Documentation Status| |Code Coverage| |License Type| |Project Stars|

----

Features
--------

Thanks to the underlying ``pylearn2`` implementation, this library supports the following neural network features, which are exposed in an intuitive and `well documented <http://scikit-neuralnetwork.readthedocs.org/>`_ API:

* **Activation Functions —**
    * Nonlinear: ``Sigmoid``, ``Tanh``, ``Rectifier``, ``Maxout``.
    * Linear: ``Linear``, ``Gaussian``, ``Softmax``.
* **Layer Types —** ``Convolution`` (greyscale and color, 2D), ``Dense`` (standard, 1D).
* **Learning Rules —** ``sgd``, ``momentum``, ``nesterov``, ``adadelta``, ``adagrad``, ``rmsprop``.
* **Regularization —** ``L1``, ``L2`` and ``dropout``.
* **Dataset Formats —** ``numpy.ndarray``, ``scipy.sparse``, coming soon: iterators.

If a feature you need is missing, consider opening a `GitHub Issue <https://github.com/aigamedev/scikit-neuralnetwork/issues>`_ with a detailed explanation about the use case and we'll see what we can do.


Installation & Testing
----------------------

Download Latest Release
~~~~~~~~~~~~~~~~~~~~~~~

If you want to use the latest official release, you can do so from PYPI directly::

    > pip install scikit-neuralnetwork

This contains its own packaged version of ``pylearn2`` from the date of the release (and tag) but will use any globally installed version if available.

Pulling From Repository
~~~~~~~~~~~~~~~~~~~~~~~

You'll need to first install some dependencies manually.  Unfortunately, ``pylearn2`` isn't yet installable via PyPI and recommends an editable (``pip -e``) installation::

    > pip install numpy scipy theano
    > pip install -e git+https://github.com/lisa-lab/pylearn2.git#egg=Package

Once that's done, you can grab this repository and install from ``setup.py`` in the exact same way::

    > git clone https://github.com/aigamedev/scikit-neuralnetwork.git
    > cd scikit-neuralnetwork; python setup.py develop
    
This will make the ``sknn`` package globally available within Python as a reference to the current directory.

Running Automated Tests
~~~~~~~~~~~~~~~~~~~~~~~

Then, you can run the samples and benchmarks available in the ``examples/`` folder, or launch the tests to check everything is working::

    > pip install nose
    > nosetests -v sknn.tests

.. image:: docs/console_tests.png

We strive to maintain 100% test coverage for all code-paths, to ensure that rapid changes in the underlying ``pylearn2`` library are caught automatically.


Demonstration
-------------

To run a visualization that uses the ``sknn.mlp.Classifier`` just run the following command in the project's root folder::

    > python examples/plot_mlp.py --params activation

There are multiple parameters you can plot as well, for example ``iterations``, ``rules`` or ``units``.  The datasets are randomized each time, but the output should be an image that looks like this...

.. image:: docs/plot_activation.png


Benchmarks
----------

The following section compares ``nolearn`` (and ``lasagne``) vs. ``sknn`` (and ``pylearn2``) by evaluating them as a black box.  In theory, these neural network models are all the same, but in practice every implementation detail can impact the result.  Here we attempt to measure the differences in the underlying libraries.

The results shown are from training for 10 epochs for two-thirds of the original MNIST data, on two different machines:

1. **GPU Results**: NVIDIA GeForce GTX 650 (Memory: 1024Mb, Cores: 384) on Ubuntu 14.04.
2. **CPU Results**: Intel Core i7 2Ghz (256kb L2, 6MB L3) on OSX Mavericks 10.9.5.

You can run the following command to reproduce the benchmarks on your machine::

    > python examples/bench_mnist.py (sknn|lasagne)

... to generate the statistics below (e.g. over 25 runs).

==========  ==================  =========================  ==================  =========================
   MNIST      sknn.mlp (CPU)      nolearn.lasagne (CPU)      sknn.mlp (GPU)      nolearn.lasagne (GPU)
==========  ==================  =========================  ==================  =========================
 Accuracy    **97.99%±0.046**          97.77% ±0.054        **98.00%±0.06**         97.76% ±0.06
 Training     **20.1s ±1.07**            45.7s ±1.10          33.10s ±0.11         **31.93s ±0.09**
==========  ==================  =========================  ==================  =========================

All the neural networks were setup as similarly as possible, given parameters that can be controlled within the implementation and their interfaces.  In particular, this model has a single hidden layer with 300 hidden units of type Rectified Linear (ReLU) and trained with the same data with validation and monitoring disabled.  The remaining third of the MNIST dataset was only used to test the score once training terminated.

**WARNING**: These numbers should not be considered definitive and fluctuate as the underlying libraries change.  If you have any ideas how to make the accuracy results similar, then please submit a Pull Request on the benchmark script.


Getting Started
---------------

The library supports both regressors (to estimate continuous outputs from inputs) and classifiers (to predict labels from features).  This is the ``sklearn``-compatible API:

.. code:: python

    from sknn.mlp import Classifier, Layer

    nn = Classifier(
        layers=[
            Layer("Rectifier", units=100),
            Layer("Linear")],
        learning_rate=0.02,
        n_iter=10)
    nn.fit(X_train, y_train)

    y_valid = nn.predict(X_valid)

    score = nn.score(X_test, y_test)

The `generated documentation <http://scikit-neuralnetwork.readthedocs.org/>`_ as a standalone page where you can find more information about parameters, as well as examples in the `User Guide <http://scikit-neuralnetwork.readthedocs.org/en/latest/guide.html>`_.


Links & References
------------------

* `PyLearn2 <https://github.com/lisa-lab/pylearn2>`_ by LISA Lab — The amazing neural network library that powers ``sknn``.
* `Theano <https://github.com/Theano/Theano>`_ by LISA Lab — Underlying array/math library for efficient computation.
* `scikit-learn <http://scikit-learn.org/>`_ by INRIA — Machine learning library with an elegant Pythonic interface.
* `nolearn <https://github.com/dnouri/nolearn>`_ by dnouri — Similar wrapper library for Lasagne compatible with ``scikit-learn``.
* `Lasagne <https://github.com/Lasagne/Lasagne>`_ by benanne — Alternative deep learning implementation using ``Theano`` too.

----

|Build Status| |Documentation Status| |Code Coverage| |License Type| |Project Stars|

.. |Build Status| image:: https://travis-ci.org/aigamedev/scikit-neuralnetwork.svg?branch=master
   :target: https://travis-ci.org/aigamedev/scikit-neuralnetwork

.. |Documentation Status| image:: https://readthedocs.org/projects/scikit-neuralnetwork/badge/?version=latest
    :target: http://scikit-neuralnetwork.readthedocs.org/

.. |Code Coverage| image:: https://coveralls.io/repos/aigamedev/scikit-neuralnetwork/badge.svg?branch=master
    :target: https://coveralls.io/r/aigamedev/scikit-neuralnetwork?branch=master

.. |License Type| image:: https://img.shields.io/badge/license-New%20BSD-blue.svg
    :target: https://github.com/aigamedev/scikit-neuralnetwork/blob/master/LICENSE

.. |Project Stars| image:: https://img.shields.io/github/stars/aigamedev/scikit-neuralnetwork.svg
    :target: https://github.com/aigamedev/scikit-neuralnetwork/stargazers    
