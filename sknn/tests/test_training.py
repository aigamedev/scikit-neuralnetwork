import unittest
from nose.tools import (assert_is_not_none, assert_raises, assert_equal)

import numpy
from sknn.mlp import BaseMLP


class TestTrainingProcedure(unittest.TestCase):

    def setUp(self):
        self.nn = BaseMLP(layers=[("LinearGaussian",)], learning_rate=0.001,
                          n_iter=None, n_stable=1, f_stable=0.1)

    def test_FitTerminateStable(self):
        a_in, a_out = numpy.zeros((8,16)), numpy.zeros((8,4))
        self.nn._fit(a_in, a_out)
