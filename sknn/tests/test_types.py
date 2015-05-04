import unittest
from nose.tools import (assert_is_not_none, assert_raises, assert_equal)

import numpy
import scipy.sparse
from sknn.mlp import MultiLayerPerceptron as MLP
from sknn.mlp import Layer as L, Convolution as C


# Sparse matrix must support indexing.  Other types but these do not work for this reason.
SPARSE_TYPES = ['csr_matrix', 'csc_matrix', 'dok_matrix', 'lil_matrix']


class TestScipySparseMatrix(unittest.TestCase):

    def setUp(self):
        self.nn = MLP(layers=[L("Gaussian", units=4)], n_iter=1)

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

    def test_Predict64(self):
        for t in SPARSE_TYPES:
            sparse_matrix = getattr(scipy.sparse, t)
            X = sparse_matrix((8, 4), dtype=numpy.float64)
            self.nn._predict(X)

    def test_Predict32(self):
        for t in SPARSE_TYPES:
            sparse_matrix = getattr(scipy.sparse, t)
            X = sparse_matrix((8, 4), dtype=numpy.float32)
            self.nn._predict(X)


class TestConvolutionError(unittest.TestCase):

    def setUp(self):
        self.nn = MLP(layers=[C("Rectifier", channels=4), L("Linear")], n_iter=1)

    def test_FitError(self):
        # The sparse matrices can't store anything but 2D, but convolution needs 3D or more.
        for t in SPARSE_TYPES:
            sparse_matrix = getattr(scipy.sparse, t)
            X, y = sparse_matrix((8, 16)), sparse_matrix((8, 16))
            assert_raises(TypeError, self.nn._fit, X, y)
