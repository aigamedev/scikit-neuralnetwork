import unittest
from nose.tools import (assert_is_not_none, assert_true, assert_raises, assert_equal)

import numpy
from sklearn.base import clone

from sknn.mlp import MultiLayerPerceptronClassifier as MLPC


class TestClassifierFunctionality(unittest.TestCase):

    def setUp(self):
        self.nn = MLPC(layers=[("Linear",)], n_iter=1)

    def test_FitAutoInitialize(self):
        a_in, a_out = numpy.zeros((8,16)), numpy.zeros((8,), dtype=numpy.int32)
        self.nn.fit(a_in, a_out)
        assert_true(self.nn.is_initialized)

    def test_PartialFit(self):
        a_in, a_out = numpy.zeros((8,4)), numpy.zeros((8,), dtype=numpy.int32)
        self.nn.partial_fit(a_in, a_out, classes=[0,1,2,3])
        self.nn.partial_fit(a_in*2.0, a_out+1, classes=[0,1,2,3])

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

    def test_CalculateScore(self):
        a_in, a_out = numpy.zeros((8,16)), numpy.zeros((8,), dtype=numpy.int32)
        self.nn.fit(a_in, a_out)
        f = self.nn.score(a_in, a_out)
        assert_equal(type(f), numpy.float64)


class TestClassifierClone(TestClassifierFunctionality):

    def setUp(self):
        cc = MLPC(layers=[("Linear",)], n_iter=1)
        self.nn = clone(cc)

    # This runs the same tests on the clone as for the original above.


class TestClassifierInterface(unittest.TestCase):

    def check_values(self, params):
        assert_equal(params['learning_rate'], 0.05)
        assert_equal(params['n_iter'], 456)
        assert_equal(params['n_stable'], 123)
        assert_equal(params['dropout'], True)
        assert_equal(params['valid_size'], 0.2)

    def test_GetParamValues(self):
        nn = MLPC(layers=[("Linear",)], learning_rate=0.05, n_iter=456,
                  n_stable=123, valid_size=0.2, dropout=True)
        params = nn.get_params()
        self.check_values(params)

    def test_CloneWithValues(self):
        nn = MLPC(layers=[("Linear",)], learning_rate=0.05, n_iter=456,
                  n_stable=123, valid_size=0.2, dropout=True)
        cc = clone(nn)
        params = cc.get_params()
        self.check_values(params)

    def check_defaults(self, params):
        assert_equal(params['learning_rate'], 0.01)
        assert_equal(params['n_iter'], None)
        assert_equal(params['n_stable'], 50)
        assert_equal(params['dropout'], False)
        assert_equal(params['valid_size'], 0.0)

    def test_GetParamDefaults(self):
        nn = MLPC(layers=[("Gaussian",)])
        params = nn.get_params()
        self.check_defaults(params)

    def test_CloneDefaults(self):
        nn = MLPC(layers=[("Gaussian",)])
        cc = clone(nn)
        params = cc.get_params()
        self.check_defaults(params)

    def test_ConvertToString(self):
        nn = MLPC(layers=[("Gaussian",)])
        assert_equal(str, type(str(nn)))

    def test_Representation(self):
        nn = MLPC(layers=[("Gaussian",)])
        assert_equal(str, type(repr(nn)))
