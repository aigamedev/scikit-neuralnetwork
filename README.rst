scikit-neuralnetwork
====================

Deep neural network implementation without the learning cliff!  This is a wrapper for the powerful ``pylearn2`` library that's compatible with ``scikit-learn`` for a more user-friendly and Pythonic interface.

|Build Status| |Documentation Status| |Code Coverage|

----

Installation
------------

You'll need to first install some dependencies manually.  Unfortunately, ``pylearn2`` isn't yet installable via PyPI and recommends an editable (``pip -e``) installation::

    > git clone https://github.com/lisa-lab/pylearn2.git
    > cd pylearn2; python setup.py develop

Once that's done, you can grab this repository and set your ``PYTHONPATH`` to point to the correct folder.  A ``setup.py`` file is coming soon for the official version 0.1!


Upcoming Features
-----------------

* Allow using all layer types as hidden layers, not linear only for output.
* Better error checking for the layer specifications, useful messages otherwise.
* Use pylearn2's monitoring code to stop training upon detecting convergence.
* Improve the classification code by using more specialized softmax activation.


.. |Build Status| image:: https://travis-ci.org/aigamedev/scikit-neuralnetwork.svg?branch=master
   :target: https://travis-ci.org/aigamedev/scikit-neuralnetwork

.. |Documentation Status| image:: https://readthedocs.org/projects/scikit-neuralnetwork/badge/?version=latest
    :target: http://scikit-neuralnetwork.readthedocs.org/

.. |Code Coverage| image:: https://coveralls.io/repos/aigamedev/scikit-neuralnetwork/badge.svg?branch=master
    :target: https://coveralls.io/r/aigamedev/scikit-neuralnetwork?branch=master
