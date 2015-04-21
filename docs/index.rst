Welcome to scikit-neuralnetwork's documentation!
================================================

Deep neural network implementation without the learning cliff! This library implements multi-layer perceptrons as a wrapper for the powerful pylearn2 library that's compatible with scikit-learn for a more user-friendly and Pythonic interface.

|Build Status| |Documentation Status| |Code Coverage|

----

Modules
-------

.. toctree::
    :maxdepth: 2

    mlp


Installation
------------

You'll need to first install some dependencies manually.  Unfortunately, ``pylearn2`` isn't yet installable via PyPI and recommends an editable (``pip -e``) installation::

    > pip install numpy scipy theano
    > pip install -e git+https://github.com/lisa-lab/pylearn2.git#egg=Package

Once that's done, you can grab this repository and install from ``setup.py`` in the exact same way::

    > git clone https://github.com/aigamedev/scikit-neuralnetwork.git
    > cd scikit-neuralnetwork; python setup.py develop

With that done, you can run the samples and benchmarks available in the ``examples/`` folder.


Getting Started
---------------

The library supports both regressors (to estimate continuous outputs) and classifiers (to predict classes).  This is the ``sklearn``-compatible API:

.. code:: python

    import sknn.mlp

    nn = sknn.mlp.MultiLayerPerceptronClassifier(
        layers=[("Rectifier", 100), ("Linear",)],
        learning_rate=0.02,
        n_iter=10)

    nn.fit(X_train, y_train)
    nn.predict(X_test)

    nn.score(X_valid, y_valid)

You can also use a ``MultiLayerPerceptronRegressor`` in the exact same way.  See the documentation in :mod:`sknn.mlp` for details about the construction parameters.


Indices & Search
----------------

* :ref:`genindex`
* :ref:`search`


.. |Build Status| image:: https://travis-ci.org/aigamedev/scikit-neuralnetwork.svg?branch=master
   :target: https://travis-ci.org/aigamedev/scikit-neuralnetwork

.. |Documentation Status| image:: https://readthedocs.org/projects/scikit-neuralnetwork/badge/?version=latest
    :target: http://scikit-neuralnetwork.readthedocs.org/

.. |Code Coverage| image:: https://coveralls.io/repos/aigamedev/scikit-neuralnetwork/badge.svg?branch=master
    :target: https://coveralls.io/r/aigamedev/scikit-neuralnetwork?branch=master
