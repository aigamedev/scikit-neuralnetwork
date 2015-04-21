import unittest
from nose.tools import (assert_in)

import io
import logging

import numpy
import sknn.mlp


class TestTrainingProcedure(unittest.TestCase):

    def test_FitTerminateStable(self):
        a_in, a_out = numpy.zeros((8,16)), numpy.zeros((8,4))
        self.nn = sknn.mlp.BaseMLP(
                    layers=[("LinearGaussian",)], learning_rate=0.001,
                    n_iter=None, n_stable=1, f_stable=0.1,
                    valid_set=(a_in, a_out))

        self.nn._fit(a_in, a_out)


class TestTrainingOutput(unittest.TestCase):

    def setUp(self):
        self.buf = io.StringIO()
        self.hnd = logging.StreamHandler(self.buf)
        logging.getLogger('sknn').addHandler(self.hnd)

    def tearDown(self):
        sknn.mlp.log.removeHandler(self.hnd)

    def test_VerboseRegressor(self):
        nn = sknn.mlp.MultiLayerPerceptronRegressor(layers=[("Linear",)], verbose=1, n_iter=1)
        a_in, a_out = numpy.zeros((8,16)), numpy.zeros((8,4))
        nn.fit(a_in, a_out)
        assert_in("Epoch    Validation Error    Time", self.buf.getvalue())
        assert_in("    0           N/A          0.0s", self.buf.getvalue())

    def test_VerboseClassifier(self):
        nn = sknn.mlp.MultiLayerPerceptronClassifier(layers=[("Linear",)], verbose=1, n_iter=1)
        a_in, a_out = numpy.zeros((8,16)), numpy.zeros((8,1), dtype=numpy.int32)
        nn.fit(a_in, a_out)
        assert_in("Epoch    Validation Error    Time", self.buf.getvalue())
        assert_in("    0           N/A          0.0s", self.buf.getvalue())
