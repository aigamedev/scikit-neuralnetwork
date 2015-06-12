# -*- coding: utf-8 -*-
from __future__ import (absolute_import, unicode_literals, print_function)

__all__ = ['AutoEncoder', 'Layer']

import time
import logging
import itertools

log = logging.getLogger('sknn')


import sklearn

from .pywrap2 import (autoencoder, sgd, transformer_dataset, blocks, ae_costs, corruption)
from .nn import NeuralNetwork


class AutoEncoder(NeuralNetwork):

    def __init__(self, spec):
        super(AutoEncoder, self).__init__(spec)
        self.dca = None

    def _ae_get_cost(self, layer):
        if layer.cost == 'msre':
            return ae_costs.MeanSquaredReconstructionError()
        if layer.cost == 'mbce':
            return ae_costs.MeanBinaryCrossEntropy()

    def _fit_impl(self, X):
        input_size = [X.shape[1]] + [l.units for l in self.layers[:-1]]
        ae_layers = []
        for v, l in zip(input_size, self.layers):
            ae_layers.append(self._create_ae_layer(v, l))
        self.dca = autoencoder.DeepComposedAutoencoder(ae_layers)

        input_space = self._create_input_space(X)
        ds = self._create_dataset(input_space, X=X)
        datasets = self._create_ae_datasets(ds, ae_layers)
        trainers = [self._create_trainer(d, self._ae_get_cost(l)) for d, l in zip(datasets, self.layers)]
        for l, t, d in zip(ae_layers, trainers, datasets):
            t.setup(l, d)
            self._train_layer(t, l, d)
        return self

    def _transform_impl(self, X):
        assert self.dca is not None, "The auto-encoder has not been trained yet."
        return self.dca.perform(X)

    def _transfer_impl(self, nn):
        nn.weights = []
        for a in self.dca.autoencoders:
            nn.weights.append((a.weights.get_value(), a.hidbias.get_value()))

    def _create_ae_layer(self, size, layer):
        """Construct an internal pylearn2 layer based on the requested layer type.
        """
        activation = layer.activation.lower()
        if layer.type == 'autoencoder':
            return autoencoder.Autoencoder(size,
                                           layer.units,
                                           activation,
                                           activation,
                                           layer.tied_weights,
                                           rng=self.random_state)
        if layer.type == 'denoising':
            corruptor = corruption.GaussianCorruptor(layer.corruption_level,
                                                     self.random_state)
            return autoencoder.DenoisingAutoencoder(corruptor,
                                                    size,
                                                    layer.units,
                                                    activation,
                                                    activation,
                                                    tied_weights=layer.tied_weights,
                                                    rng=self.random_state)

    def _create_ae_datasets(self, ds, layers):
        """Setup pylearn2 transformer datasets for each layer, passing the inputs through
        previous layers before training the current layer.
        """
        trainsets = [ds]
        for i, l in enumerate(layers):
            stack = layers[0] if i == 0 else blocks.StackedBlocks(layers[0:i+1])
            trds = transformer_dataset.TransformerDataset(raw=ds, transformer=stack)
            trainsets.append(trds)
        return trainsets
