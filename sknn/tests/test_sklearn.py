import unittest
from nose.tools import (assert_equal, assert_raises, assert_in, assert_not_in)

import numpy
from scipy.stats import randint, uniform

from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import VotingClassifier
    
from sknn.mlp import Regressor as MLPR, Classifier as MLPC
from sknn.mlp import Layer as L


class TestGridSearchRegressor(unittest.TestCase):
    
    __estimator__ = MLPR
    __output__ = "Linear"
    
    def setUp(self):
        self.a_in = numpy.random.uniform(0.0, 1.0, (64,16))
        self.a_out = numpy.zeros((64,1))

    def test_GridGlobalParams(self):
        clf = GridSearchCV(
                    self.__estimator__(layers=[L(self.__output__)], n_iter=1),
                    param_grid={'learning_rate': [0.01, 0.001]})
        clf.fit(self.a_in, self.a_out)

    def test_GridLayerParams(self):
        clf = GridSearchCV(
                    self.__estimator__(layers=[L("Rectifier", units=12), L(self.__output__)], n_iter=1),
                    param_grid={'hidden0__units': [4, 8, 12]})
        clf.fit(self.a_in, self.a_out)

    def test_RandomGlobalParams(self):
        clf = RandomizedSearchCV(
                    self.__estimator__(layers=[L("Sigmoid")], n_iter=1),
                    param_distributions={'learning_rate': uniform(0.001, 0.01)},
                    n_iter=2)
        clf.fit(self.a_in, self.a_out)

    def test_RandomLayerParams(self):
        clf = RandomizedSearchCV(
                    self.__estimator__(layers=[L("Rectifier", units=12), L(self.__output__)], n_iter=1),
                    param_distributions={'hidden0__units': randint(4, 12)},
                    n_iter=2)
        clf.fit(self.a_in, self.a_out)

    def test_RandomMultipleJobs(self):
        clf = RandomizedSearchCV(
                    self.__estimator__(layers=[L("Sigmoid", units=12), L(self.__output__)], n_iter=1),
                    param_distributions={'hidden0__units': randint(4, 12)},
                    n_iter=4, n_jobs=4)
        clf.fit(self.a_in, self.a_out)


class TestGridSearchClassifier(TestGridSearchRegressor):
    # NOTE: "multiclass-multioutput is not supported" by sklearn.

    __estimator__ = MLPC
    __output__ = "Softmax"

    def setUp(self):
        self.a_in = numpy.random.uniform(0.0, 1.0, (64,16))
        self.a_out = numpy.random.randint(0, 4, (64,))


class TestCrossValidation(unittest.TestCase):

    def test_Regressor(self):
        a_in = numpy.random.uniform(0.0, 1.0, (64,16))
        a_out = numpy.zeros((64,1))

        cross_val_score(MLPR(layers=[L("Linear")], n_iter=1), a_in, a_out, cv=5)

    def test_Classifier(self):
        a_in = numpy.random.uniform(0.0, 1.0, (64,16))
        a_out = numpy.random.randint(0, 4, (64,))

        cross_val_score(MLPC(layers=[L("Softmax")], n_iter=1), a_in, a_out, cv=5)


class TestVotingEnsemble(unittest.TestCase):
    
    def test_SingleVote(self):
        a_in, a_out = numpy.random.uniform(0.0, 1.0, (64,16)), numpy.zeros((64,))
        vc = VotingClassifier([('nn1', MLPC(layers=[L("Softmax")], n_iter=1))])
        vc.fit(a_in, a_out)
        vc.predict(a_in)
