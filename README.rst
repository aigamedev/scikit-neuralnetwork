scikit-neuralnetwork
====================

Deep neural network implementation without the learning cliff!  This library implements multi-layer perceptrons as a wrapper for the powerful ``pylearn2`` library that's compatible with ``scikit-learn`` for a more user-friendly and Pythonic interface. Oh, and it runs on your GPU by default.

**NOTE**: This project is possible thanks to the `nucl.ai Conference <http://nucl.ai/>`_ on **July 20-22**. Join us in **Vienna**!

|Build Status| |Documentation Status| |Code Coverage|

----

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

To run a visualization that uses the ``sknn.mlp.MultiLayerPerceptronClassifier`` just run the following command in the project's root folder::

    > python examples/plot_mlp.py --params activation

There are multiple parameters you can plot as well, for example ``iterations``, ``rules`` or ``units``.  The datasets are randomized each time, but the output should be an image that looks like this...

.. image:: docs/plot_activation.png


Benchmarks
----------

Here are the results of testing 10 epochs of training for two-thirds of the original MNIST data, on Ubuntu 14.04 and a GeForce GTX 650 (Memory: 1024Mb, Cores: 384).  You can run the following command::

    > python examples/bench_mnist.py (sknn|lasagne)

... to generate the results below.

.. class:: center

==========  ============  ===============  ===================
   MNIST      sknn.mlp      nolearn.dbn      nolearn.lasagne
==========  ============  ===============  ===================
 Accuracy    **98.05%**       97.80%             97.78%
 Training        36s           274s              **32s**
==========  ============  ===============  ===================

All the networks have a single hidden layer with 300 hidden units of the default type, and were given the same data with validation and monitoring disabled.  The remaining third of the MNIST dataset was only used to test the score once training terminated.

**WARNING**: For the ``theano`` powered libraries, these numbers are somewhat sensitive to parameter changes so please do not consider them definitive!  It's likely tweaking parameters in both libraries would make training times very similar...


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

The `generated documentation <http://scikit-neuralnetwork.readthedocs.org/>`_ as a standalone page where you can find more information about parameters, as well as examples in the User Guide.


.. |Build Status| image:: https://travis-ci.org/aigamedev/scikit-neuralnetwork.svg?branch=master
   :target: https://travis-ci.org/aigamedev/scikit-neuralnetwork

.. |Documentation Status| image:: https://readthedocs.org/projects/scikit-neuralnetwork/badge/?version=latest
    :target: http://scikit-neuralnetwork.readthedocs.org/

.. |Code Coverage| image:: https://coveralls.io/repos/aigamedev/scikit-neuralnetwork/badge.svg?branch=master
    :target: https://coveralls.io/r/aigamedev/scikit-neuralnetwork?branch=master
