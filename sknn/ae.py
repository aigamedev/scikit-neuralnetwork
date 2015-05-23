# -*- coding: utf-8 -*-
from __future__ import (absolute_import, unicode_literals, print_function)

__all__ = ['AutoEncoder', 'Layer']

import time
import logging
import itertools

log = logging.getLogger('sknn')


import sklearn

from .pywrap2 import (autoencoder, sgd, transformer_dataset, blocks, ae_costs, corruption)
from . import nn


class Layer(nn.Layer):
    """
    Specification for a layer to be passed to the auto-encoder during construction.  This
    includes a variety of parameters to configure each layer based on its activation type.

    Parameters
    ----------

    activation: str
        Select which activation function this layer should use, as a string.  Specifically,
        options are ``Sigmoid`` and ``Tanh`` only for such auto-encoders.

    type: str, optional
        The type of encoding and decoding layer to use, specifically ``denoising`` for randomly
        corrupting data, and a more traditional ``autoencoder`` which is used by default.

    name: str, optional
        You optionally can specify a name for this layer, and its parameters
        will then be accessible to scikit-learn via a nested sub-object.  For example,
        if name is set to ``layer1``, then the parameter ``layer1__units`` from the network
        is bound to this layer's ``units`` variable.

        The name defaults to ``hiddenN`` where N is the integer index of that layer, and the
        final layer is always ``output`` without an index.

    units: int
        The number of units (also known as neurons) in this layer.  This applies to all
        layer types except for convolution.

    cost: string, optional
        What type of cost function to use during the layerwise pre-training.  This can be either
        ``msre`` for mean-squared reconstruction error (default), and ``mbce`` for mean binary
        cross entropy.

    tied_weights: bool, optional
        Whether to use the same weights for the encoding and decoding phases of the simulation
        and training.  Default is ``True``.

    corruption_level: float, optional
        The ratio of inputs to corrupt in this layer; ``0.25`` means that 25% of the inputs will be
        corrupted during the training.  The default is ``0.5``.

    warning: None
        You should use keyword arguments after `type` when initializing this object. If not,
        the code will raise an AssertionError.
    """

    def __init__(self,
                 activation,
                 warning=None,
                 type='autoencoder',
                 name=None,
                 units=None,
                 cost='msre',
                 tied_weights=True,
                 corruption_level=0.5):

        assert warning is None, \
            "Specify layer parameters as keyword arguments, not positional arguments."

        if type not in ['denoising', 'autoencoder']:
            raise NotImplementedError("AutoEncoder layer type `%s` is not implemented." % type)
        if cost not in ['msre', 'mbce']:
            raise NotImplementedError("Error type '%s' is not implemented." % cost)
        if activation not in ['Sigmoid', 'Tanh']:
            raise NotImplementedError("Activation type '%s' is not implemented." % activation)

        self.activation = activation
        self.type = type
        self.name = name
        self.units = units
        self.cost = cost
        self.tied_weights = tied_weights
        self.corruption_level = corruption_level

    def _get_cost(self):
        if self.cost == 'msre':
            return ae_costs.MeanSquaredReconstructionError()
        if self.cost == 'mbce':
            return ae_costs.MeanBinaryCrossEntropy()


class AutoEncoder(nn.NeuralNetwork, sklearn.base.TransformerMixin):

    def _setup(self):
        self.dca = None

    def fit(self, X):
        """Fit the auto-encoder to the given data using layerwise training.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_inputs)
            Training vectors as real numbers, where ``n_samples`` is the number of
            samples and ``n_inputs`` is the number of input features.

        Returns
        -------
        self : object
            Returns this instance.
        """
        sgd.log.setLevel(logging.WARNING)
        num_samples, data_size = X.shape[0], X.size

        log.info("Training on dataset of {:,} samples with {:,} total size.".format(num_samples, data_size))
        if self.n_iter:
            log.debug("  - Terminating loop after {} total iterations.".format(self.n_iter))
        if self.n_stable:
            log.debug("  - Early termination after {} stable iterations.".format(self.n_stable))

        if self.verbose:
            log.debug("\nEpoch    Validation Error    Time"
                      "\n---------------------------------")

        input_size = [X.shape[1]] + [l.units for l in self.layers[:-1]]
        ae_layers = []
        for v, l in zip(input_size, self.layers):
            ae_layers.append(self._create_ae_layer(v, l))

        ds, _ = self._create_matrix_input(X=X, y=None)
        datasets = self._create_ae_datasets(ds, ae_layers)
        trainers = [self._create_trainer(d, l._get_cost()) for d, l in zip(datasets, self.layers)]
        for l, t, d in zip(ae_layers, trainers, datasets):
            t.setup(l, d)
            self._train_layer(t, l, d)

        self.dca = autoencoder.DeepComposedAutoencoder(ae_layers)
        return self

    def transform(self, X):
        """Encode the data via the neural network, as an upward pass simulation to
        generate high-level features from the low-level inputs.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_inputs)
            Input data to be passed through the auto-encoder upward.

        Returns
        -------
        y : numpy array, shape (n_samples, n_features)
            Transformed output array from the auto-encoder.
        """
        assert self.dca is not None, "The auto-encoder has not been trained yet."
        return self.dca.perform(X)

    def transfer(self, nn):
        for a, l in zip(self.layers, nn.layers):
            assert a.activation == l.type,\
                "Mismatch in activation types in target MLP; expected `%s` but found `%s`."\
                % (a.activation, l.type)
            assert a.units == l.units,\
                "Different number of units in target MLP; expected `%i` but found `%i`."\
                % (a.units, l.units)

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
