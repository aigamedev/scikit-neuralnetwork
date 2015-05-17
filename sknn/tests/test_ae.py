import unittest
from nose.tools import (assert_false, assert_raises, assert_true,
                        assert_equal, assert_in)

import numpy

from sknn.ae import AutoEncoder as AE, Layer as L


class TestAutoEncoder(unittest.TestCase):
    
    def test_LifeCycle(self):
        ae = AE(layers=[L("Sigmoid", units=8)])
        del ae

    def test_FitData(self):
        X = numpy.zeros((8,4))
        ae = AE(layers=[L("Sigmoid", units=8)], n_iter=1)
        ae.fit(X)
