import unittest

import io
import pickle
import numpy as np

from nose.tools import (assert_is_not_none)
from sklearn.utils.testing import (assert_raises, assert_equal)

from sknn import NeuralNetwork

from . import test_linear

class TestDeepNetwork(test_linear.TestLinearNetwork):

    def setUp(self):
        self.nn = NeuralNetwork(
            layers=[
                ("RectifiedLinear", 16),
                ("Sigmoid", 12),
                ("Maxout", 8, 2),
                ("Tanh", 4),
                ("Linear",),
            ])
