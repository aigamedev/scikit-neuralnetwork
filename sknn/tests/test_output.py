import unittest
from nose.tools import (assert_is_not_none, assert_raises, assert_equal)

from sknn.mlp import Regressor as MLPR
from sknn.mlp import Layer as L

from . import test_linear


class TestGaussianOutput(test_linear.TestLinearNetwork):

    def setUp(self):
        self.nn = MLPR(layers=[L("Gaussian")], n_iter=1)


class TestSoftmaxOutput(test_linear.TestLinearNetwork):

    def setUp(self):
        self.nn = MLPR(layers=[L("Softmax")], n_iter=1)
