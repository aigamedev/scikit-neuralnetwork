import unittest
from nose.tools import (assert_raises)

from sknn.nn import NeuralNetwork as NN


class TestAbstractNeuralNetwork(unittest.TestCase):

    def test_SetupRaisesException(self):
        assert_raises(NotImplementedError, NN, layers=[])

    def test_IsInitializedRaisesExecption(self):
        assert_raises(NotImplementedError, NN.is_initialized.fget, object())
