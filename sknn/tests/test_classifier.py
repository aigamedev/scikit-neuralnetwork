import unittest
from nose.tools import (assert_is_not_none, assert_true, assert_raises,
                        assert_in, assert_equal, assert_less_equal)

import numpy
from sklearn.base import clone

from sknn.mlp import Classifier as MLPC
from sknn.mlp import Layer as L, Convolution as C


class TestClassifierFunctionality(unittest.TestCase):

    def setUp(self):
        self.nn = MLPC(layers=[L("Softmax")], n_iter=1)

    def test_IsClassifier(self):
        assert_true(self.nn.is_classifier)

    def test_FitAutoInitialize(self):
        a_in, a_out = numpy.zeros((8,16)), numpy.random.randint(0, 5, (8,))
        self.nn.fit(a_in, a_out)
        assert_true(self.nn.is_initialized)

    def test_ExplicitValidSet(self):
        a_in, a_out = numpy.zeros((8,16)), numpy.random.randint(0, 5, (8,))
        self.nn.valid_set = (a_in, a_out)
        self.nn.fit(a_in, a_out)
        assert_true(self.nn.is_initialized)

    def test_PartialFit(self):
        a_in, a_out = numpy.zeros((8,4)), numpy.random.randint(0, 5, (8,))
        self.nn.partial_fit(a_in, a_out, classes=[0,1,2,3])
        self.nn.partial_fit(a_in*2.0, a_out+1, classes=[0,1,2,3])

    def test_PredictUninitializedNoUnitCount(self):
        a_in = numpy.zeros((8,16))
        assert_raises(AssertionError, self.nn.predict, a_in)

    def test_PredictUninitializedNoLabels(self):
        self.nn.layers[-1].units = 4
        a_in = numpy.zeros((8,16))
        assert_raises(AssertionError, self.nn.predict, a_in)

    def test_PredictBinaryProbability(self):
        a_in = numpy.random.uniform(-1.0, 1.0, size=(8,16))
        a_out = numpy.array((a_in.sum(axis=1) >= 0.0), dtype=numpy.int32)
        a_out[0], a_out[-1] = 0, 1
        self.nn.fit(a_in, a_out)

        a_proba = self.nn.predict_proba(a_in)
        a_test = self.nn.predict(a_in)
        c_out = numpy.unique(a_out)

        assert_equal(2, c_out.shape[0])
        assert_equal((8, 2), a_proba.shape)

        assert_true((a_proba >= 0.0).all())
        assert_true((a_proba <= 1.0).all())
        assert_true((abs(a_proba.sum(axis=1) - 1.0) < 1E-9).all())

    def test_PredictClasses(self):
        a_in, a_out = numpy.zeros((8,16)), numpy.random.randint(0, 5, (8,))
        self.nn.fit(a_in, a_out)
        self.nn.batch_size = 4
        a_test = self.nn.predict(a_in)
        assert_equal(type(a_out), type(a_test))
        assert_equal(a_out.shape[0], a_test.shape[0])

        c_out = numpy.unique(a_out)
        assert_equal(len(self.nn.classes_), 1)
        assert_true((self.nn.classes_[0] == c_out).all())

    def test_PredictLargerBatchSize(self):
        a_in, a_out = numpy.zeros((8,16)), numpy.random.randint(0, 5, (8,1))
        self.nn.batch_size = 32

        self.nn.fit(a_in, a_out)
        a_test = self.nn.predict(a_in)
        assert_equal(type(a_out), type(a_test))
        assert_equal(a_out.shape[0], a_test.shape[0])

    def test_PredictMultiClass(self):
        a_in, a_out = numpy.zeros((32,16)), numpy.random.randint(0, 3, (32,2))
        self.nn.fit(a_in, a_out)
        a_test = self.nn.predict(a_in)
        assert_equal(type(a_out), type(a_test))
        assert_equal(a_out.shape, a_test.shape)

        assert_equal(len(self.nn.classes_), 2)
        assert_equal(self.nn.classes_[0].shape[0], 3)
        assert_equal(self.nn.classes_[1].shape[0], 3)

    def test_EstimateProbalities(self):
        a_in, a_out = numpy.zeros((8,16)), numpy.random.randint(0, 5, (8,))
        self.nn.fit(a_in, a_out)
        a_proba = self.nn.predict_proba(a_in)
        assert_equal(type(a_out), type(a_proba))
        assert_equal(a_in.shape[0], a_proba.shape[0])

        assert_true((a_proba >= 0.0).all())
        assert_true((a_proba <= 1.0).all())
        assert_true((abs(a_proba.sum(axis=1) - 1.0) < 1E-9).all())

    def test_MultipleProbalitiesAsList(self):
        a_in, a_out = numpy.zeros((8,16)), numpy.random.randint(0, 5, (8,4))
        self.nn.fit(a_in, a_out)
        a_proba = self.nn.predict_proba(a_in)
        assert_equal(list, type(a_proba))
        assert_equal(4, len(a_proba))

        for p in a_proba:
            assert_equal(a_in.shape[0], p.shape[0])
            assert_less_equal(p.shape[1], 5)
            assert_true((p >= 0.0).all())
            assert_true((p <= 1.0).all())
            assert_true((abs(p.sum(axis=1) - 1.0) < 1E-9).all())

    def test_CalculateScore(self):
        a_in, a_out = numpy.zeros((8,16)), numpy.random.randint(0, 5, (8,))
        self.nn.fit(a_in, a_out)
        f = self.nn.score(a_in, a_out)
        assert_equal(type(f), numpy.float64)


class TestClassifierClone(TestClassifierFunctionality):

    def setUp(self):
        cc = MLPC(layers=[L("Sigmoid")], n_iter=1)
        self.nn = clone(cc)

    # This runs the same tests on the clone as for the original above.


class TestClassifierInterface(unittest.TestCase):

    def check_values(self, params):
        assert_equal(params['learning_rate'], 0.05)
        assert_equal(params['n_iter'], 456)
        assert_equal(params['n_stable'], 123)
        assert_equal(params['dropout_rate'], 0.25)
        assert_equal(params['regularize'], 'dropout')
        assert_equal(params['valid_size'], 0.2)

    def test_GetParamValues(self):
        nn = MLPC(layers=[L("Linear")], learning_rate=0.05, n_iter=456,
                  n_stable=123, valid_size=0.2, dropout_rate=0.25)
        params = nn.get_params()
        self.check_values(params)

    def test_CloneWithValues(self):
        nn = MLPC(layers=[L("Linear")], learning_rate=0.05, n_iter=456,
                  n_stable=123, valid_size=0.2, dropout_rate=0.25)
        cc = clone(nn)
        params = cc.get_params()
        self.check_values(params)

    def check_defaults(self, params):
        assert_equal(params['learning_rate'], 0.01)
        assert_equal(params['n_iter'], None)
        assert_equal(params['n_stable'], 10)
        assert_equal(params['regularize'], None)
        assert_equal(params['valid_size'], 0.0)

    def test_GetParamDefaults(self):
        nn = MLPC(layers=[L("Gaussian")])
        params = nn.get_params()
        self.check_defaults(params)

    def test_CloneDefaults(self):
        nn = MLPC(layers=[L("Gaussian")])
        cc = clone(nn)
        params = cc.get_params()
        self.check_defaults(params)

    def test_ConvertToString(self):
        nn = MLPC(layers=[L("Gaussian")])
        assert_equal(str, type(str(nn)))

    def test_RepresentationDenseLayer(self):
        nn = MLPC(layers=[L("Gaussian")])
        r = repr(nn)
        assert_equal(str, type(r))
        assert_in("sknn.nn.Layer `Gaussian`", r)

    def test_RepresentationConvolution(self):
        nn = MLPC(layers=[C("Rectifier")])
        r = repr(nn)
        assert_equal(str, type(r))
        assert_in("sknn.nn.Convolution `Rectifier`", r)
