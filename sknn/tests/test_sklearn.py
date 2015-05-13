import unittest
from nose.tools import (assert_equal, assert_raises, assert_in, assert_not_in)

import numpy
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score

from sknn.mlp import Regressor as MLPR
from sknn.mlp import Layer as L


class TestGridSearch(unittest.TestCase):

    def test_RegressorGlobalParams(self):
        a_in = numpy.random.uniform(0.0, 1.0, (64,16))
        a_out = numpy.zeros((64,1))

        clf = GridSearchCV(
                    MLPR(layers=[L("Linear")], n_iter=1),
                    param_grid={'learning_rate': [0.01, 0.001]})
        clf.fit(a_in, a_out)

    def test_RegressorLayerParams(self):
        a_in = numpy.random.uniform(0.0, 1.0, (64,16))
        a_out = numpy.zeros((64,1))

        clf = GridSearchCV(
                    MLPR(layers=[L("Rectifier", units=12), L("Linear")], n_iter=1),
                    param_grid={'hidden0__units': [4, 8, 12]})
        clf.fit(a_in, a_out)


class TestCrossValidation(unittest.TestCase):

    def test_Regressor(self):
        a_in = numpy.random.uniform(0.0, 1.0, (64,16))
        a_out = numpy.zeros((64,1))

        cross_val_score(MLPR(layers=[L("Linear")], n_iter=1), a_in, a_out, cv=5)
