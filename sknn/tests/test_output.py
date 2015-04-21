import unittest
from nose.tools import (assert_is_not_none, assert_raises, assert_equal)

from sknn.mlp import MultiLayerPerceptronRegressor as MLPR, MultiLayerPerceptronClassifier as MLPC

from . import test_linear


class TestGaussianOutput(test_linear.TestLinearNetwork):

    def setUp(self):
        self.nn = MLPR(layers=[("Gaussian",)], n_iter=1)


class TestSoftmaxOutput(test_linear.TestLinearNetwork):

    def setUp(self):
        self.nn = MLPC(layers=[("Softmax",)], n_iter=1)
