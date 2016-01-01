import unittest
from nose.tools import (assert_is_not_none, assert_true, assert_raises, assert_equal)

import io
import pickle
import numpy

from sknn.mlp import Regressor as MLPR
from sknn.mlp import Layer as L, Convolution as C


class TestConvolution(unittest.TestCase):

    def _run(self, nn, a_in=None, fit=True):
        assert_true(nn.is_convolution())
        if a_in is None:
            a_in = numpy.zeros((8,32,16,1))
        a_out = numpy.zeros((8,4))
        if fit is True:
            nn.fit(a_in, a_out)
        a_test = nn.predict(a_in)
        assert_equal(type(a_out), type(a_in))
        return a_test

    def test_MissingLastDim(self):
        self._run(MLPR(
            layers=[
                C("Tanh", channels=4, kernel_shape=(3,3)),
                L("Linear")],
            n_iter=1),
            a_in=numpy.zeros((8,32,16)))

    def test_SquareKernel(self):
        self._run(MLPR(
            layers=[
                C("Rectifier", channels=4, kernel_shape=(3,3)),
                L("Linear")],
            n_iter=1))

    def test_KernelPooling(self):
        self._run(MLPR(
            layers=[
                C("Rectifier", channels=4, kernel_shape=(3,3), pool_shape=(2,2)),
                L("Linear")],
            n_iter=1))

    def test_VerticalKernel(self):
        self._run(MLPR(
            layers=[
                C("Rectifier", channels=4, kernel_shape=(16,1), border_mode='valid'),
                L("Linear")],
            n_iter=1))

    def test_VerticalVerbose(self):
        self._run(MLPR(
            layers=[
                C("Sigmoid", channels=4, kernel_shape=(16,1), border_mode='valid'),
                L("Linear")],
            n_iter=1, verbose=1, valid_size=0.1))

    def test_HorizontalKernel(self):
        self._run(MLPR(
            layers=[
                C("Rectifier", channels=4, kernel_shape=(1,16), border_mode='valid'),
                L("Linear")],
            n_iter=1))

    def test_ValidationSize(self):
        self._run(MLPR(
            layers=[
                C("Tanh", channels=4, kernel_shape=(3,3)),
                L("Linear")],
            n_iter=1,
            valid_size=0.5))

    def test_ValidationSet(self):
        v_in = numpy.zeros((8,32,16,1))
        v_out = numpy.zeros((8,4))

        self._run(MLPR(
            layers=[
                C("Tanh", channels=4, kernel_shape=(3,3)),
                L("Linear")],
            n_iter=1,
            valid_set=(v_in, v_out)))

    def test_MultipleLayers(self):
        self._run(MLPR(
            layers=[
                C("Rectifier", channels=6, kernel_shape=(3,3)),
                C("Sigmoid", channels=4, kernel_shape=(5,5)),
                C("Tanh", channels=8, kernel_shape=(3,3)),
                L("Linear")],
            n_iter=1))

    def test_PoolingMaxType(self):
        self._run(MLPR(
            layers=[
                C("Rectifier", channels=4, kernel_shape=(2,2),
                               pool_shape=(2,2), pool_type='max'),
                L("Linear")],
            n_iter=1))

    def test_PoolingMeanType(self):
        self._run(MLPR(
            layers=[
                C("Rectifier", channels=4, kernel_shape=(2,2),
                               pool_shape=(2,2), pool_type='mean'),
                L("Linear")],
            n_iter=1))



class TestUpscaling(unittest.TestCase):

    def _run(self, nn, scale):
        assert_true(nn.is_convolution())
        a_in = numpy.zeros((8,16,16,1))
        a_out = numpy.zeros((8,16*scale,16*scale,3))
        nn.fit(a_in, a_out)
        a_test = nn.predict(a_in)
        assert_equal(type(a_out), type(a_in))
        return a_test

    def test_UpscalingFactorFour(self):
        self._run(MLPR(
                    layers=[
                        C("Rectifier", channels=3, kernel_shape=(3,3), scale_factor=(4,4), border_mode='same')],
                    n_iter=1),
                  scale=4) 
    
    def test_DownscaleUpscale(self):
        self._run(MLPR(
                    layers=[
                        C("ExpLin", channels=6, kernel_shape=(3,3), pool_shape=(2,2), border_mode='same'),
                        C("Rectifier", channels=3, kernel_shape=(3,3), scale_factor=(2,2), border_mode='same')],
                    n_iter=1),
                  scale=1)


class TestConvolutionSpecs(unittest.TestCase):

    def test_SmallSquareKernel(self):
        nn = MLPR(layers=[
                    C("Rectifier", channels=4, kernel_shape=(3,3), border_mode='valid'),
                    L("Linear", units=5)])

        a_in = numpy.zeros((8,32,32,1))
        nn._create_specs(a_in)
        assert_equal(nn.unit_counts, [1024, 30 * 30 * 4, 5])

    def test_SquareKernelFull(self):
        nn = MLPR(layers=[
                    C("ExpLin", channels=4, kernel_shape=(3,3), border_mode='full'),
                    L("Linear", units=5)])

        a_in = numpy.zeros((8,32,32,1))
        nn._create_specs(a_in)
        assert_equal(nn.unit_counts, [1024, 4624, 5])

    def test_HorizontalKernel(self):
        nn = MLPR(layers=[
                    C("Rectifier", channels=7, kernel_shape=(16,1), border_mode='valid'),
                    L("Linear", units=5)])

        a_in = numpy.zeros((8,16,16,1))
        nn._create_specs(a_in)
        assert_equal(nn.unit_counts, [256, 16 * 7, 5])

    def test_VerticalKernel(self):
        nn = MLPR(layers=[
                    C("Rectifier", channels=4, kernel_shape=(1,16), border_mode='valid'),
                    L("Linear", units=7)])

        a_in = numpy.zeros((8,16,16,1))
        nn._create_specs(a_in)
        assert_equal(nn.unit_counts, [256, 16 * 4, 7])

    def test_SquareKernelPool(self):
        nn = MLPR(layers=[
                    C("ExpLin", channels=4, kernel_shape=(3,3), pool_shape=(2,2), border_mode='valid'),
                    L("Linear", units=5)])

        a_in = numpy.zeros((8,32,32,1))
        nn._create_specs(a_in)
        assert_equal(nn.unit_counts, [1024, 15 * 15 * 4, 5])

    def test_SquarePoolFull(self):
        nn = MLPR(layers=[
                    C("Rectifier", channels=4, kernel_shape=(3,3), pool_shape=(2,2), border_mode='full'),
                    L("Linear", units=5)])

        a_in = numpy.zeros((8,32,32,1))
        nn._create_specs(a_in)
        assert_equal(nn.unit_counts, [1024, 16 * 16 * 4, 5])

    def test_InvalidBorderMode(self):
        assert_raises(NotImplementedError, C,
                      "Rectifier", channels=4, kernel_shape=(3,3), border_mode='unknown')

    def test_MultiLayerPooling(self):
        nn = MLPR(layers=[
                    C("Rectifier", channels=4, kernel_shape=(3,3), pool_shape=(2,2)),
                    C("ExpLin", channels=4, kernel_shape=(3,3), pool_shape=(2,2)),
                    L("Linear")])

        a_in, a_out = numpy.zeros((8,32,32,1)), numpy.zeros((8,16))
        nn._initialize(a_in, a_out)
        assert_equal(nn.unit_counts, [1024, 900, 196, 16])

    def test_Upscaling(self):
        nn = MLPR(layers=[
                    C("Rectifier", channels=4, kernel_shape=(1,1), scale_factor=(2,2), border_mode='same'),
                    L("Linear", units=5)])

        a_in = numpy.zeros((8,32,32,1))
        nn._create_specs(a_in)
        assert_equal(nn.unit_counts, [1024, 64 * 64 * 4, 5])


class TestActivationTypes(unittest.TestCase):

    def _run(self, activation):
        a_in, a_out = numpy.zeros((8,32,16,1)), numpy.zeros((8,4))
        nn = MLPR(
            layers=[
                C(activation, channels=4, kernel_shape=(3,3),
                              pool_shape=(2,2), pool_type='mean'),
                L("Linear")],
            n_iter=1)
        nn.fit(a_in, a_out)
        a_test = nn.predict(a_in)
        assert_equal(type(a_out), type(a_in))

    def test_RectifierConv(self):
        self._run("Rectifier")

    def test_ExponentialLinear(self):
        self._run("ExpLin")

    def test_SigmoidConv(self):
        self._run("Sigmoid")

    def test_TanhConv(self):
        self._run("Tanh")

    def test_LinearConv(self):
        self._run("Linear")

    def test_UnknownConv(self):
        assert_raises(NotImplementedError, self._run, "Unknown")


class TestConvolutionRGB(TestConvolution):

    def _run(self, nn, a_in=None):
        if a_in is None:
            a_in = numpy.zeros((8,32,16,1))
        a_out = numpy.zeros((8,4))

        nn.fit(a_in, a_out)
        a_test = nn.predict(a_in)
        assert_equal(type(a_out), type(a_in))


class TestSerialization(unittest.TestCase):

    def setUp(self):
        self.nn = MLPR(
            layers=[
                C("Rectifier", channels=6, kernel_shape=(3,3)),
                C("Sigmoid", channels=4, kernel_shape=(5,5)),
                C("Tanh", channels=8, kernel_shape=(3,3)),
                L("Linear")],
            n_iter=1)

    def test_SerializeEmpty(self):
        buf = io.BytesIO()
        pickle.dump(self.nn, buf)
        buf.seek(0)
        nn = pickle.load(buf)

        buf = io.BytesIO()
        pickle.dump(self.nn, buf)

        buf.seek(0)
        nn = pickle.load(buf)

        assert_equal(nn.layers, self.nn.layers)


class TestSerializedNetwork(TestConvolution):

    def _run(self, original, a_in=None):
        a_test = super(TestSerializedNetwork, self)._run(original, a_in)

        buf = io.BytesIO()
        pickle.dump(original, buf)
        buf.seek(0)
        nn = pickle.load(buf)

        a_copy = super(TestSerializedNetwork, self)._run(nn, a_in, fit=False)
        assert_true((a_test == a_copy).all())
