Installation
============

Downloading Package
-------------------

To download and setup the last officially released package, you can do so from PYPI directly::

    > pip install scikit-neuralnetwork

This contains its own packaged version of ``pylearn2`` from the date of the release (and tag) but will use any globally installed version if available.

If you want to install the very latest from source, please visit the `Project Page <http://github.com/aigamedev/scikit-neuralnetwork>`_ on GitHub for details.


Running Tests
-------------

We encourage you to launch the tests to check everything is working using the following commands::

    > pip install nose
    > nosetests -v sknn

Use the additional command-line parameters in the test runner ``--processes=8`` and ``--process-timeout=60`` to speed things up on powerful machines.  The result should look as follows in your terminal.

.. image:: console_tests.png

We strive to maintain 100% test coverage for all code-paths, to ensure that rapid changes in the underlying ``pylearn2`` library are caught automatically.

