# -*- coding: utf-8 -*-
from __future__ import (absolute_import, unicode_literals, print_function)

__all__ = ['AutoEncoder', 'Layer']

import time
import logging
import itertools

log = logging.getLogger('sknn')


import sklearn

from . import nn, backend


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


class AutoEncoder(nn.NeuralNetwork, sklearn.base.TransformerMixin):

    def _setup(self):
        assert not self.is_initialized,\
            "This auto-encoder has already been initialized."

        backend.setup()
        self._backend = backend.AutoEncoderBackend(self)

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
        num_samples, data_size = X.shape[0], X.size

        log.info("Training on dataset of {:,} samples with {:,} total size.".format(num_samples, data_size))
        if self.n_iter:
            log.debug("  - Terminating loop after {} total iterations.".format(self.n_iter))
        if self.n_stable:
            log.debug("  - Early termination after {} stable iterations.".format(self.n_stable))

        if self.verbose:
            log.debug("\nEpoch    Validation Error        Time"
                      "\n-------------------------------------")
        
        self._backend._fit_impl(X)
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
        return self._backend._transform_impl(X)

    def transfer(self, nn):
        assert not nn.is_initialized,\
            "Target multi-layer perceptron has already been initialized."

        for a, l in zip(self.layers, nn.layers):
            assert a.activation == l.type,\
                "Mismatch in activation types in target MLP; expected `%s` but found `%s`."\
                % (a.activation, l.type)
            assert a.units == l.units,\
                "Different number of units in target MLP; expected `%i` but found `%i`."\
                % (a.units, l.units)
       
        self._backend._transfer_impl(nn)
