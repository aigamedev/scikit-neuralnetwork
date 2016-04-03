Installation
============

You have multiple options to get up and running, though using ``pip`` is by far the easiest and most reliable.

A) Download Latest Release [Recommended]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you want to use the latest official release, you can do so from PYPI directly::

    > pip install scikit-neuralnetwork

This will install the latest official ``Lasagne`` and ``Theano`` as well as other minor packages too as a dependency.  We strongly suggest you use a `virtualenv <https://virtualenv.pypa.io/en/latest/>`_ for Python.

B) Pulling Repositories [Optional]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you want to use the more advanced features like convolution, pooling or upscaling, these depend on the latest code from ``Lasagne`` and ``Theano`` master branches.  You can install them manually as follows::

    > pip install -r https://raw.githubusercontent.com/aigamedev/scikit-neuralnetwork/master/requirements.txt

Once that's done, you can grab this repository and install from ``setup.py`` in the exact same way::

    > git clone https://github.com/aigamedev/scikit-neuralnetwork.git
    > cd scikit-neuralnetwork; python setup.py develop
    
This will make the ``sknn`` package globally available within Python as a reference to the current directory.


Running Tests
-------------

We encourage you to launch the tests to check everything is working using the following commands::

    > pip install nose
    > nosetests -v sknn

Use the additional command-line parameters in the test runner ``--processes=8`` and ``--process-timeout=60`` to speed things up on powerful machines.  The result should look as follows in your terminal.

.. image:: console_tests.png

We strive to maintain 100% test coverage for all code-paths, to ensure that rapid changes in the underlying ``Lasagne`` and ``Theano`` libraries are caught automatically.
