import unittest

import io
import pickle
import numpy as np

from nose.tools import (assert_is_not_none, assert_raises, assert_equal)

from sknn.mlp import MultiLayerPerceptronRegressor as MLPR

from . import test_linear


class TestDeepNetwork(test_linear.TestLinearNetwork):

    def setUp(self):
        self.nn = MLPR(
            layers=[
                ("RectifiedLinear", 16),
                ("Sigmoid", 12),
                ("Maxout", 8, 2),
                ("Tanh", 4),
                ("Linear",),
            ])
