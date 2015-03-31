import numpy as np

from sknn import NeuralNetwork


def test_instantiation():
    nn = NeuralNetwork(layers=[("Linear",)])

def test_prediction():
    nn = NeuralNetwork(layers=[("Linear",)])
    a_in = np.zeros((8,16))
    nn.predict(a_in, 4)