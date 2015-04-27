import unittest
from nose.tools import (assert_is_not_none, assert_raises, assert_equal)

import scipy.sparse
from sknn.mlp import BaseMLP as MLP
from sknn.mlp import Layer as L


class TestInputDataTypes(unittest.TestCase):

    def setUp(self):
        self.nn = MLP(layers=[L("Gaussian")], n_iter=1)

    def test_FitSciPySparse(self):
        X, y = scipy.sparse.csr_matrix((8, 4)), scipy.sparse.csr_matrix((8, 4))
        self.nn._fit(X, y)

    def test_PredictSciPySparse(self):
        X, y = scipy.sparse.csr_matrix((8, 4)), scipy.sparse.csr_matrix((8, 4))
        self.nn._fit(X, y)
        self.nn._predict(X)

