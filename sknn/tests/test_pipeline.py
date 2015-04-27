import unittest
from nose.tools import (assert_equal, assert_true)

import io
import pickle
import numpy

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from sknn.mlp import Regressor as MLPR
from sknn.mlp import Layer as L


class TestPipeline(unittest.TestCase):

    def _run(self, pipeline):
        a_in, a_out = numpy.zeros((8,16)), numpy.zeros((8,4))
        pipeline.fit(a_in, a_out)
        pipeline.predict(a_in)

    def test_NeuralNetworkOnly(self):
        pipeline = Pipeline([
            ('neural network', MLPR(layers=[L("Linear")], n_iter=1))
        ])
        self._run(pipeline)

    def test_ScalerThenNeuralNetwork(self):
        pipeline = Pipeline([
            ('min/max scaler', MinMaxScaler()),
            ('neural network', MLPR(layers=[L("Linear")], n_iter=1))
        ])
        self._run(pipeline)


class TestSerializedPipeline(TestPipeline):

    def _run(self, pipeline):
        a_in, a_out = numpy.zeros((8,16)), numpy.zeros((8,4))
        pipeline.fit(a_in, a_out)        
        a_test = pipeline.predict(a_in)

        buf = io.BytesIO()
        pickle.dump(pipeline, buf)

        buf.seek(0)
        p = pickle.load(buf)
        
        assert_true((a_test == p.predict(a_in)).all())
