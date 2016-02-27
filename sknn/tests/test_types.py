import unittest
from nose.tools import (assert_is_not_none, assert_raises, assert_equal, assert_true)

import os
import random
import shutil
import tempfile

import numpy
import pandas
import theano
import scipy.sparse

from sknn.mlp import MultiLayerPerceptron as MLP
from sknn.mlp import Layer as L, Convolution as C


# Sparse matrix must support indexing.  Other types but these do not work for this reason.
SPARSE_TYPES = ['csr_matrix', 'csc_matrix', 'dok_matrix', 'lil_matrix']


class TestScipySparseMatrix(unittest.TestCase):

    def setUp(self):
        self.nn = MLP(layers=[L("Linear", units=4)], n_iter=1)

    def test_FitFloat64(self):
        for t in SPARSE_TYPES:
            sparse_matrix = getattr(scipy.sparse, t)
            X = sparse_matrix((8, 4), dtype=numpy.float64)
            y = sparse_matrix((8, 4), dtype=numpy.float64)
            self.nn._fit(X, y)

    def test_FitFloat32(self):
        for t in SPARSE_TYPES:
            sparse_matrix = getattr(scipy.sparse, t)
            X = sparse_matrix((8, 4), dtype=numpy.float32)
            y = sparse_matrix((8, 4), dtype=numpy.float32)
            self.nn._fit(X, y)

    def test_FitHybrid(self):
        for t in SPARSE_TYPES:
            sparse_matrix = getattr(scipy.sparse, t)
            X = sparse_matrix((8, 4), dtype=numpy.float32)
            y = numpy.zeros((8, 4), dtype=numpy.float32)
            self.nn._fit(X, y)

    def test_FitMutator(self):
        def mutate(Xb, **_):
            self.count += 1
            Xb -= 0.5
        self.nn.callback = {'on_batch_start': mutate}

        for t in SPARSE_TYPES:
            sparse_matrix = getattr(scipy.sparse, t)
            X = sparse_matrix((8, 4), dtype=numpy.float32)
            y = numpy.zeros((8, 4), dtype=numpy.float32)

            self.count = 0
            assert_equal(0, self.count)
            self.nn._fit(X, y)
            assert_equal(8, self.count)

    def test_Predict64(self):
        theano.config.floatX = 'float64'
        for t in SPARSE_TYPES:
            sparse_matrix = getattr(scipy.sparse, t)
            X = sparse_matrix((8, 4), dtype=numpy.float64)
            yp = self.nn._predict(X)
            assert_equal(yp.dtype, numpy.float64)

    def test_Predict32(self):
        theano.config.floatX = 'float32'
        for t in SPARSE_TYPES:
            sparse_matrix = getattr(scipy.sparse, t)
            X = sparse_matrix((8, 4), dtype=numpy.float32)
            yp = self.nn._predict(X)
            assert_equal(yp.dtype, numpy.float32)


class TestMemoryMap(unittest.TestCase):

    __types__ = ['float32', 'float64']

    def setUp(self):
        self.nn = MLP(layers=[L("Linear", units=3)], n_iter=1)
        self.directory = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.directory)

    def make(self, name, shape, dtype):
        filename = os.path.join(self.directory, name)
        return numpy.memmap(filename, dtype=dtype, mode='w+', shape=shape)

    def test_FitAllTypes(self):
        for t in self.__types__:
            theano.config.floatX = t
            X = self.make('X', (12, 3), dtype=t)
            y = self.make('y', (12, 3), dtype=t)
            self.nn._fit(X, y)

    def test_PredictAllTypes(self):
        for t in self.__types__:
            theano.config.floatX = t
            X = self.make('X', (12, 3), dtype=t)
            yp = self.nn._predict(X)


class TestPandasDataFrame(TestMemoryMap):
    
    __types__ = ['float32']

    def make(self, _, shape, dtype):
        return pandas.DataFrame(numpy.random.uniform(-1.0, 1.0, size=shape), dtype=dtype)


class TestConvolution(unittest.TestCase):

    def setUp(self):
        self.nn = MLP(
            layers=[
                C("Rectifier", kernel_shape=(3,3), channels=4),
                L("Linear")],
            n_iter=1)

    def test_FitError(self):
        # The sparse matrices can't store anything but 2D, but convolution needs 3D or more.
        for t in SPARSE_TYPES:
            sparse_matrix = getattr(scipy.sparse, t)
            X, y = sparse_matrix((8, 16)), sparse_matrix((8, 16))
            assert_raises((TypeError, NotImplementedError), self.nn._fit, X, y)

    def test_FitResizeSquare(self):
        # The sparse matrices can't store anything but 2D, but convolution needs 3D or more.
        X, y = numpy.zeros((8, 36)), numpy.zeros((8, 4))
        self.nn._fit(X, y)

    def test_FitResizeFails(self):
        # The sparse matrices can't store anything but 2D, but convolution needs 3D or more.
        X, y = numpy.zeros((8, 35)), numpy.zeros((8, 4))
        assert_raises(AssertionError, self.nn._fit, X, y)


class TestFormatDeterminism(unittest.TestCase):

    def test_TrainRandomOneEpoch(self):
        for t in ['dok_matrix', 'lil_matrix']:
            sparse_matrix = getattr(scipy.sparse, t)
            X_s, y_s = sparse_matrix((8, 16), dtype=numpy.float32), sparse_matrix((8, 16), dtype=numpy.float32)
            for i in range(X_s.shape[0]):
                X_s[i,random.randint(0, X_s.shape[1]-1)] = 1.0
                y_s[i,random.randint(0, y_s.shape[1]-1)] = 1.0
            X, y = X_s.toarray(), y_s.toarray()

            nn1 = MLP(layers=[L("Linear")], n_iter=1, random_state=1234)
            nn1._fit(X, y)

            nn2 = MLP(layers=[L("Linear")], n_iter=1, random_state=1234)
            nn2._fit(X_s, y_s)

            assert_true(numpy.all(nn1._predict(X_s) == nn1._predict(X_s)))

    def test_TrainConstantOneEpoch(self):
        for t in ['csr_matrix', 'csc_matrix']:
            sparse_matrix = getattr(scipy.sparse, t)
            X_s, y_s = sparse_matrix((8, 16), dtype=numpy.float32), sparse_matrix((8, 16), dtype=numpy.float32)
            X, y = X_s.toarray(), y_s.toarray()
            
            nn1 = MLP(layers=[L("Linear")], n_iter=1, random_state=1234)
            nn1._fit(X, y)

            nn2 = MLP(layers=[L("Linear")], n_iter=1, random_state=1234)
            nn2._fit(X_s, y_s)

            assert_true(numpy.all(nn1._predict(X_s) == nn1._predict(X_s)))
