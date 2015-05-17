import time
import logging
import itertools

import sklearn

from pylearn2.models import autoencoder
from pylearn2.datasets.transformer_dataset import TransformerDataset
from pylearn2.blocks import StackedBlocks
from pylearn2.costs import autoencoder as ae_costs
from pylearn2.corruption import GaussianCorruptor

from . import nn

log = logging.getLogger('sknn')


class Layer(nn.Layer):

    def __init__(self,
                 activation,
                 warning=None,
                 type='autoencoder',
                 name=None,
                 units=None,
                 cost='msre',
                 tied_weights=False,
                 corruption_level=0.5):

        assert warning is None, \
            "Specify layer parameters as keyword arguments, not positional arguments."

        if type not in ['denoising', 'autoencoder']:
            raise NotImplementedError("AutoEncoder layer type `%s` is not implemented." % type)
        if cost not in ['msre', 'mbce']:
            raise NotImplementedError("Error type '%s' is not implemented." % cost)

        self.name = name
        self.type = type
        self.units = units
        self.activation = activation.lower()
        self.tied_weights = tied_weights
        self.corruption_level = corruption_level
        self.cost = cost

    def get_cost(self):
        if self.cost == 'msre':
            return ae_costs.MeanSquaredReconstructionError()
        if self.cost == 'mbce':
            return ae_costs.MeanBinaryCrossEntropy()


class AutoEncoder(nn.NeuralNetwork, sklearn.base.TransformerMixin):

    def _setup(self):
        pass

    def fit(self, X):
        input_size = [X.shape[1]] + [l.units for l in self.layers[:-1]]
        ae_layers = []
        for v, l in zip(input_size, self.layers):
            ae_layers.append(self._create_ae_layer(v, l))

        ds, _ = self._create_matrix_input(X=X, y=None)
        datasets = self._create_ae_datasets(ds, ae_layers)
        trainers = [self._create_trainer(d, l.get_cost()) for d, l in zip(datasets, self.layers)]
        for l, t, d in zip(ae_layers, trainers, datasets):
            t.setup(l, d)
            self._train(t, l, d)

    def _create_ae_layer(self, size, layer):
        if layer.type == 'autoencoder':
            return autoencoder.Autoencoder(size,
                                           layer.units,
                                           layer.activation,
                                           layer.activation,
                                           layer.tied_weights,
                                           rng=self.random_state)
        if layer.type == 'denoising':
            corruptor = GaussianCorruptor(layer.corruption_level, self.random_state)
            return autoencoder.DenoisingAutoencoder(corruptor,
                                                    size,
                                                    layer.units,
                                                    layer.activation,
                                                    layer.activation,
                                                    tied_weights=layer.tied_weights,
                                                    rng=self.random_state)

    def _create_ae_datasets(self, ds, layers):
        trainsets = [ds]
        for i, l in enumerate(layers):
            stack = layers[0] if i == 0 else StackedBlocks(layers[0:i+1])
            trds = TransformerDataset(raw=ds, transformer=stack)
            trainsets.append(trds)
        return trainsets
