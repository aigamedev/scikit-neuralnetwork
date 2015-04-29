Welcome to sknn's documentation!
================================

Deep neural network implementation without the learning cliff! This library implements multi-layer perceptrons as a wrapper for the powerful ``pylearn2`` library that's compatible with ``scikit-learn`` for a more user-friendly and Pythonic interface.

|Build Status| |Documentation Status| |Code Coverage| |License| |Source Code|

----

Module Reference
----------------

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

Then, you can run the samples and benchmarks available in the ``examples/`` folder.


Running Tests
-------------

We encourage you to launch the tests to check everything is working using the following commands::

    > pip install nose
    > nosetests -v sknn

Use the additional command-line parameters in the test runner ``--processes=8`` and ``--process-timeout=60`` to speed things up on powerful machines.  The result should look as follows in your terminal.

.. image:: console_tests.png

We strive to maintain 100% test coverage for all code-paths, to ensure that rapid changes in the underlying ``pylearn2`` library are caught automatically.


Getting Started
---------------

.. toctree::
    :maxdepth: 2

    guide

.. image:: plot_activation.png


Indices & Search
----------------

* :ref:`genindex`
* :ref:`search`


----

|Build Status| |Documentation Status| |Code Coverage| |License Type| |Source Code|

.. |Build Status| image:: https://travis-ci.org/aigamedev/scikit-neuralnetwork.svg?branch=master
   :target: https://travis-ci.org/aigamedev/scikit-neuralnetwork

.. |Documentation Status| image:: https://readthedocs.org/projects/scikit-neuralnetwork/badge/?version=latest
    :target: http://scikit-neuralnetwork.readthedocs.org/

.. |Code Coverage| image:: https://coveralls.io/repos/aigamedev/scikit-neuralnetwork/badge.svg?branch=master
    :target: https://coveralls.io/r/aigamedev/scikit-neuralnetwork?branch=master

.. |License Type| image:: https://img.shields.io/badge/license-New%20BSD-blue.svg
    :target: https://github.com/aigamedev/scikit-neuralnetwork/blob/master/LICENSE

.. |Source Code| image:: https://img.shields.io/github/stars/aigamedev/scikit-neuralnetwork.svg
    :target: https://github.com/aigamedev/scikit-neuralnetwork/
