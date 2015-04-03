"""
test_clone
test_repr
test_str
test_getparams
test_isclassifier
test_setparams
test_fit
test_score
"""

import unittest
from nose.tools import (assert_is_not_none, assert_true, assert_raises, assert_equal)

import numpy

from sknn.mlp import MultiLayerPerceptronClassifier as MLPC


class TestClassifier(unittest.TestCase):

    def setUp(self):
        self.nn = MLPC(layers=[("Linear",)], n_iter=1)

    def test_FitAutoInitialize(self):
        a_in, a_out = numpy.zeros((8,16)), numpy.zeros((8,), dtype=numpy.int32)
        self.nn.fit(a_in, a_out)
        assert_true(self.nn.is_initialized)

    def test_PartialFit(self):
        a_in, a_out = numpy.zeros((8,4)), numpy.zeros((8,), dtype=numpy.int32)
        self.nn.partial_fit(a_in, a_out)
        self.nn.partial_fit(a_in*2.0, a_out+1)

    def test_PredictUninitialized(self):
        a_in = numpy.zeros((8,16))
        assert_raises(ValueError, self.nn.predict, a_in)

    def test_PredictClasses(self):
        a_in, a_out = numpy.zeros((8,16)), numpy.zeros((8,), dtype=numpy.int32)
        self.nn.fit(a_in, a_out)
        a_test = self.nn.predict(a_in)
        assert_equal(type(a_out), type(a_test))
        assert_equal(a_out.shape, a_test.shape)

    def test_EstimateProbalities(self):
        a_in, a_out = numpy.zeros((8,16)), numpy.zeros((8,), dtype=numpy.int32)
        self.nn.fit(a_in, a_out)
        a_test = self.nn.predict_proba(a_in)
        assert_equal(type(a_out), type(a_test))
