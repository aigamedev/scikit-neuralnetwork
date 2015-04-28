scikit-neuralnetwork
====================

Deep neural network implementation without the learning cliff!  This library implements multi-layer perceptrons as a wrapper for the powerful ``pylearn2`` library that's compatible with ``scikit-learn`` for a more user-friendly and Pythonic interface. Oh, and it runs on your GPU by default.

**NOTE**: This project is possible thanks to the `nucl.ai Conference <http://nucl.ai/>`_ on **July 20-22**. Join us in **Vienna**!

|Build Status| |Documentation Status| |Code Coverage|

----

Features
--------

Thanks to the underlying ``PyLearn2`` implementation, this library supports the following: 

* **Activation Types** —
    * Nonlinear: ``Sigmoid``, ``Tanh``, ``Rectifier``, ``Maxout``.
    * Linear: ``Linear``, ``Gaussian``, ``Softmax``.
* **Layer Types** — ``Convolution`` (greyscale and color), ``Feed Forward`` (standard).
* **Learning Rules** — ``sgd``, ``nesterov``, ``adadelta``, ``adagrad``, ``rmsprop``.
* **Dataset Types** — ``numpy.ndarray``, ``scipy.sparse``, custom iterator.

If a feature you need is missing, consider opening an `Issue <https://github.com/aigamedev/scikit-neuralnetwork/issues>`_ with a detailed explanation about the use case.

Installation & Testing
----------------------

You'll need to first install some dependencies manually.  Unfortunately, ``pylearn2`` isn't yet installable via PyPI and recommends an editable (``pip -e``) installation::

    > pip install numpy scipy theano
    > pip install -e git+https://github.com/lisa-lab/pylearn2.git#egg=Package

Once that's done, you can grab this repository and install from ``setup.py`` in the exact same way::

    > git clone https://github.com/aigamedev/scikit-neuralnetwork.git
    > cd scikit-neuralnetwork; python setup.py develop

Then, you can run the samples and benchmarks available in the ``examples/`` folder, or launch the tests to check everything is working::

    > pip install nose
    > nosetests -v sknn

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

The following section compares ``nolearn`` (and ``lasagne``) with ``sknn`` (and ``pylearn2``) by evaluating them as a black box.  In theory, neural network models are all the same, but in practice every implementation detail can impact the result.  Here we attempt to 

The results shown are from training for 10 epochs for two-thirds of the original MNIST data, on Ubuntu 14.04 and a GeForce GTX 650 (Memory: 1024Mb, Cores: 384).  You can run the following command::

    > python examples/bench_mnist.py (sknn|lasagne)

... to generate the statistics below (e.g. for 25 samples).

.. class:: center

==========  ==================  =========================
   MNIST      sknn.mlp (CPU)      nolearn.lasagne (CPU)
==========  ==================  =========================
 Accuracy      97.99% ±0.046          97.77% ±0.054
 Training       20.1s ±1.07           45.70s ±1.10
==========  ==================  =========================

All the neural networks were setup as similarly as possible, given parameters that can be controlled within the implementation.  The model has a single hidden layer with 300 hidden units of type Rectified Linear (ReLU), and were given the same data with validation and monitoring disabled.  The remaining third of the MNIST dataset was only used to test the score once training terminated.

**WARNING**: These numbers are certainly not final and fluctuate as the underlying libraries change.  If you have any explanations of these scores, or ideas how to make the results similar, then please submit a Pull Request on the benchmark script!


Getting Started
---------------

The library supports both regressors (to estimate continuous outputs) and classifiers (to predict classes).  This is the ``sklearn``-compatible API:

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


.. |Build Status| image:: https://travis-ci.org/aigamedev/scikit-neuralnetwork.svg?branch=master
   :target: https://travis-ci.org/aigamedev/scikit-neuralnetwork

.. |Documentation Status| image:: https://readthedocs.org/projects/scikit-neuralnetwork/badge/?version=latest
    :target: http://scikit-neuralnetwork.readthedocs.org/

.. |Code Coverage| image:: https://coveralls.io/repos/aigamedev/scikit-neuralnetwork/badge.svg?branch=master
    :target: https://coveralls.io/r/aigamedev/scikit-neuralnetwork?branch=master
