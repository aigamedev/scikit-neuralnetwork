scikit-neuralnetwork
====================

Deep neural network implementation without the learning cliff!  This library implements multi-layer perceptrons as a wrapper for the powerful ``pylearn2`` library that's compatible with ``scikit-learn`` for a more user-friendly and Pythonic interface. Oh, and it runs on your GPU by default.

**NOTE**: This project is made possible thanks to the `nucl.ai Conference <http://nucl.ai/>`_ on **July 20-22**. Join us in **Vienna**!

|Build Status| |Documentation Status| |Code Coverage|

----

Installation
------------

You'll need to first install some dependencies manually.  Unfortunately, ``pylearn2`` isn't yet installable via PyPI and recommends an editable (``pip -e``) installation::

    > git clone https://github.com/lisa-lab/pylearn2.git
    > cd pylearn2; python setup.py develop

Once that's done, you can grab this repository and set your ``PYTHONPATH`` to point to the correct folder.  A ``setup.py`` file is coming soon for the official version 0.1!


Demonstration
-------------

To run a visualization that uses the `sknn.mlp.MultiLayerPerceptron` just run the following command::

    > PYTHONPATH=. python examples/plot_mlp.py --params activation

The datasets are randomized each time, but the output should be an image that looks like this...

.. image:: docs/plot_activation.png


Upcoming Features
-----------------

* Full tests for sklearn ``Classifier`` and ``Regressor`` compatibility.
* Quick start in the README.rst file showing how to get an estimator.
* Allow using all layer types as hidden layers, not linear only for output.
* Better error checking for the layer specifications, useful messages otherwise.
* Support for RGB images (as 3-channel input arrays) in the convolution network.


.. |Build Status| image:: https://travis-ci.org/aigamedev/scikit-neuralnetwork.svg?branch=master
   :target: https://travis-ci.org/aigamedev/scikit-neuralnetwork

.. |Documentation Status| image:: https://readthedocs.org/projects/scikit-neuralnetwork/badge/?version=latest
    :target: http://scikit-neuralnetwork.readthedocs.org/

.. |Code Coverage| image:: https://coveralls.io/repos/aigamedev/scikit-neuralnetwork/badge.svg?branch=master
    :target: https://coveralls.io/r/aigamedev/scikit-neuralnetwork?branch=master
