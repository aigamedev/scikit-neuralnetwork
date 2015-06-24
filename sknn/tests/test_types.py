import unittest
from nose.tools import (assert_is_not_none, assert_raises, assert_equal, assert_true)

import random

import theano
import numpy
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
        def mutate(x):
            x -= 0.5
            self.count += 1
        self.nn.mutator = mutate

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
