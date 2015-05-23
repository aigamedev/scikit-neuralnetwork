# -*- coding: utf-8 -*-
from __future__ import (absolute_import, unicode_literals, print_function)

__all__ = ['Regressor', 'Classifier', 'Layer', 'Convolution']

import os
import time
import logging
import itertools

log = logging.getLogger('sknn')


import numpy
import theano
import sklearn.base
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.cross_validation

from .pywrap2 import (datasets, space, sgd, mlp, maxout, cost, mlp_cost, dropout)
from .pywrap2 import learning_rule as lr, termination_criteria as tc

from .nn import NeuralNetwork, Layer, Convolution, ansi
from .dataset import SparseDesignMatrix, FastVectorSpace


class MultiLayerPerceptron(NeuralNetwork, sklearn.base.BaseEstimator):
    """
    Abstract base class for wrapping the multi-layer perceptron functionality
    from PyLearn2.
    """

    def _setup(self):
        self.unit_counts = None
        self.input_space = None
        self.mlp = None
        self.weights = None
        self.ds = None
        self.vs = None
        self.f = None
        self.trainer = None
        self.cost = None
        self.train_set = None

    def _create_mlp_trainer(self, dataset):
        sgd.log.setLevel(logging.WARNING)

        # Aggregate all the dropout parameters into shared dictionaries.
        dropout_probs, dropout_scales = {}, {}
        for l in [l for l in self.layers if l.dropout is not None]:
            incl = 1.0 - l.dropout
            dropout_probs[l.name] = incl
            dropout_scales[l.name] = 1.0 / incl
        assert len(dropout_probs) == 0 or self.regularize in ('dropout', None)

        if self.regularize == "dropout" or len(dropout_probs) > 0:
            # Use the globally specified dropout rate when there are no layer-specific ones.
            incl = 1.0 - (self.dropout_rate or 0.5)
            default_prob, default_scale = incl, 1.0 / incl

            # Pass all the parameters to pylearn2 as a custom cost function.
            self.cost = dropout.Dropout(
                default_input_include_prob=default_prob,
                default_input_scale=default_scale,
                input_include_probs=dropout_probs, input_scales=dropout_scales)

        # Aggregate all regularization parameters into common dictionaries.
        layer_decay = {}
        if self.regularize in ('L1', 'L2') or any(l.weight_decay for l in self.layers):
            wd = self.weight_decay or 0.0001
            for l in self.layers:
                layer_decay[l.name] = l.weight_decay or wd
        assert len(layer_decay) == 0 or self.regularize in ('L1', 'L2', None)

        if len(layer_decay) > 0:
            mlp_default_cost = self.mlp.get_default_cost()
            if self.regularize == 'L1':
                l1 = mlp_cost.L1WeightDecay(layer_decay)
                self.cost = cost.SumOfCosts([mlp_default_cost,l1])
            else: # Default is 'L2'.
                l2 =  mlp_cost.WeightDecay(layer_decay)
                self.cost = cost.SumOfCosts([mlp_default_cost,l2])

        return self._create_trainer(dataset, self.cost)

    def _check_layer(self, layer, required, optional=[]):
        required.extend(['name', 'type'])
        for r in required:
            if getattr(layer, r) is None:
                raise ValueError("Layer type `%s` requires parameter `%s`."\
                                 % (layer.type, r))

        optional.extend(['dropout', 'weight_decay'])
        for a in layer.__dict__:
            if a in required+optional:
                continue
            if getattr(layer, a) is not None:
                log.warning("Parameter `%s` is unused for layer type `%s`."\
                            % (a, layer.type))

    def _create_convolution_layer(self, name, layer, irange):
        self._check_layer(layer,
                          required=['channels', 'kernel_shape'],
                          optional=['kernel_stride', 'border_mode', 'pool_shape', 'pool_type'])

        if layer.type == 'Rectifier':
            nl = mlp.RectifierConvNonlinearity(0.0)
        elif layer.type == 'Sigmoid':
            nl = mlp.SigmoidConvNonlinearity()
        elif layer.type == 'Tanh':
            nl = mlp.TanhConvNonlinearity()
        else:
            assert layer.type == 'Linear',\
                "Convolution layer type `%s` is not supported." % layer.type
            nl = mlp.IdentityConvNonlinearity()

        return mlp.ConvElemwise(
            layer_name=name,
            nonlinearity=nl,
            output_channels=layer.channels,
            kernel_shape=layer.kernel_shape,
            kernel_stride=layer.kernel_stride,
            border_mode=layer.border_mode,
            pool_shape=layer.pool_shape,
            pool_type=layer.pool_type,
            pool_stride=(1,1),
            irange=irange)

    def _create_layer(self, name, layer, irange):
        if isinstance(layer, Convolution):
            return self._create_convolution_layer(name, layer, irange)

        if layer.type == 'Rectifier':
            self._check_layer(layer, ['units'])
            return mlp.RectifiedLinear(
                layer_name=name,
                dim=layer.units,
                irange=irange)

        if layer.type == 'Sigmoid':
            self._check_layer(layer, ['units'])
            return mlp.Sigmoid(
                layer_name=name,
                dim=layer.units,
                irange=irange)

        if layer.type == 'Tanh':
            self._check_layer(layer, ['units'])
            return mlp.Tanh(
                layer_name=name,
                dim=layer.units,
                irange=irange)

        if layer.type == 'Maxout':
            self._check_layer(layer, ['units', 'pieces'])
            return maxout.Maxout(
                layer_name=name,
                num_units=layer.units,
                num_pieces=layer.pieces,
                irange=irange)

        if layer.type == 'Linear':
            self._check_layer(layer, ['units'])
            return mlp.Linear(
                layer_name=layer.name,
                dim=layer.units,
                irange=irange,
                use_abs_loss=bool(self.loss_type == 'mae'))

        if layer.type == 'Gaussian':
            self._check_layer(layer, ['units'])
            return mlp.LinearGaussian(
                layer_name=layer.name,
                init_beta=0.1,
                min_beta=0.001,
                max_beta=1000,
                beta_lr_scale=None,
                dim=layer.units,
                irange=irange,
                use_abs_loss=bool(self.loss_type == 'mae'))

        if layer.type == 'Softmax':
            self._check_layer(layer, ['units'])
            return mlp.Softmax(
                layer_name=layer.name,
                n_classes=layer.units,
                irange=irange)

    def _create_mlp(self):
        mlp.logger.setLevel(logging.WARNING)

        # Create the layers one by one, connecting to previous.
        mlp_layers = []
        for i, layer in enumerate(self.layers):
            fan_in = self.unit_counts[i]
            fan_out = self.unit_counts[i + 1]

            lim = numpy.sqrt(6) / numpy.sqrt(fan_in + fan_out)
            if layer.type == 'Tanh':
                lim *= 1.1 * lim
            elif layer.type in ('Rectifier', 'Maxout'):
                # He, Rang, Zhen and Sun, converted to uniform.
                lim *= numpy.sqrt(2)
            elif layer.type == 'Sigmoid':
                lim *= 4

            mlp_layer = self._create_layer(layer.name, layer, irange=lim)
            mlp_layers.append(mlp_layer)

        log.info(
            "Initializing neural network with %i layers, %i inputs and %i outputs.",
            len(self.layers), self.unit_counts[0], self.layers[-1].units)

        self.mlp = mlp.MLP(
            mlp_layers,
            nvis=None if self.input_space else self.unit_counts[0],
            seed=self.random_state,
            input_space=self.input_space)

        for l, p, count in zip(self.layers, self.mlp.layers, self.unit_counts[1:]):
            space = p.get_output_space()
            if isinstance(l, Convolution):                
                log.debug("  - Convl: {}{: <10}{} Output: {}{: <10}{} Channels: {}{}{}".format(
                    ansi.BOLD, l.type, ansi.ENDC,
                    ansi.BOLD, repr(space.shape), ansi.ENDC,
                    ansi.BOLD, space.num_channels, ansi.ENDC))

                # NOTE: Numbers don't match up exactly for pooling; one off. The logic is convoluted!
                # assert count == numpy.product(space.shape) * space.num_channels,\
                #     "Mismatch in the calculated number of convolution layer outputs."
            else:
                log.debug("  - Dense: {}{: <10}{} Units:  {}{: <4}{}".format(
                    ansi.BOLD, l.type, ansi.ENDC, ansi.BOLD, l.units, ansi.ENDC))
                assert count == space.get_total_dimension(),\
                    "Mismatch in the calculated number of dense layer outputs."
        log.debug("")

        if self.weights is not None:
            self._array_to_mlp(self.weights, self.mlp)
            self.weights = None

        inputs = self.mlp.get_input_space().make_theano_batch()
        self.f = theano.function([inputs], self.mlp.fprop(inputs))

    def _create_specs(self, X, y=None):
        # Automatically work out the output unit count based on dataset.
        if y is not None and self.layers[-1].units is None:
            self.layers[-1].units = y.shape[1]
        else:
            assert y is None or self.layers[-1].units == y.shape[1],\
                "Mismatch between dataset size and units in output layer."

        # Then compute the number of units in each layer for initialization.
        self.unit_counts = [numpy.product(X.shape[1:]) if self.is_convolution else X.shape[1]]
        res = X.shape[1:3] if self.is_convolution else None 
        for l in self.layers:
            if isinstance(l, Convolution):
                if l.border_mode == 'valid':
                    res = (int((res[0] - l.kernel_shape[0]) / l.kernel_stride[0]) + 1,
                           int((res[1] - l.kernel_shape[1]) / l.kernel_stride[1]) + 1)
                if l.border_mode == 'full':
                    res = (int((res[0] + l.kernel_shape[0]) / l.kernel_stride[0]) - 1,
                           int((res[1] + l.kernel_shape[1]) / l.kernel_stride[1]) - 1)
                unit_count = numpy.prod(res) * l.channels
            else:
                unit_count = l.units

            self.unit_counts.append(unit_count)

    def _initialize(self, X, y):
        assert not self.is_initialized,\
            "This neural network has already been initialized."
        self._create_specs(X, y)

        if self.valid_size > 0.0:
            assert self.valid_set is None, "Can't specify valid_size and valid_set together."
            X, X_v, y, y_v = sklearn.cross_validation.train_test_split(
                                X, y,
                                test_size=self.valid_size,
                                random_state=self.random_state)
            self.valid_set = X_v, y_v
        self.train_set = X, y

        # Convolution networks need a custom input space.
        self.ds, self.input_space = self._create_matrix_input(X, y)
        if self.valid_set:
            X_v, y_v = self.valid_set
            self.vs, _ = self._create_matrix_input(X_v, y_v)
        else:
            self.vs = None

        self._create_mlp()

        self.trainer = self._create_mlp_trainer(self.vs)
        self.trainer.setup(self.mlp, self.ds)

    @property
    def is_initialized(self):
        """Check if the neural network was setup already.
        """
        return not (self.mlp is None or self.f is None)

    def __getstate__(self):
        assert self.mlp is not None,\
            "The neural network has not been initialized."

        d = self.__dict__.copy()
        d['weights'] = self._mlp_to_array()

        for k in ['ds', 'vs', 'f', 'trainer', 'mlp']:
            if k in d:
                del d[k]
        return d

    def _mlp_to_array(self):
        return [(l.get_weights(), l.get_biases()) for l in self.mlp.layers]

    def __setstate__(self, d):
        self.__dict__.update(d)
        for k in ['ds', 'vs', 'f', 'trainer', 'mlp']:
            setattr(self, k, None)
        self._create_mlp()

    def _array_to_mlp(self, array, nn):
        for layer, (weights, biases) in zip(nn.layers, array):
            assert layer.get_weights().shape == weights.shape
            layer.set_weights(weights)

            assert layer.get_biases().shape == biases.shape
            layer.set_biases(biases)

    def _reshape(self, X, y=None):
        if y is not None and y.ndim == 1:
            y = y.reshape((y.shape[0], 1))
        if self.is_convolution and X.ndim == 3:
            X = X.reshape((X.shape[0], X.shape[1], X.shape[2], 1))
        if not self.is_convolution and X.ndim > 2:
            X = X.reshape((X.shape[0], numpy.product(X.shape[1:])))
        return X, y

    def _fit(self, *data, **extra):
        try:
            return self._train(*data, **extra)
        except RuntimeError as e:
            log.error("\n{}{}{}\n\n{}\n".format(
                ansi.RED,
                "A runtime exception was caught during training. This likely occurred due to\n"
                "a divergence of the SGD algorithm, and NaN floats were found by PyLearn2.",
                ansi.ENDC,
                "Try setting the `learning_rate` 10x lower to resolve this, for example:\n"
                "    learning_rate=%f" % (self.learning_rate * 0.1)))
            raise e

    def _train(self, X, y, test=None):
        assert X.shape[0] == y.shape[0],\
            "Expecting same number of input and output samples."
        data_shape, data_size = X.shape, X.size+y.size
        X, y = self._reshape(X, y)

        if not self.is_initialized:
            self._initialize(X, y)
            X, y = self.train_set
        else:
            self.train_set = X, y
        assert self.ds is not None, "Training after serialization is not (yet) supported."

        log.info("Training on dataset of {:,} samples with {:,} total size.".format(data_shape[0], data_size))
        if data_shape[1:] != X.shape[1:]:
            log.warning("  - Reshaping input array from {} to {}.".format(data_shape, X.shape))
        if self.valid_set:
            X_v, _ = self.valid_set
            log.debug("  - Train: {: <9,}  Valid: {: <4,}".format(X.shape[0], X_v.shape[0]))
        if self.n_iter:
            log.debug("  - Terminating loop after {} total iterations.".format(self.n_iter))
        if self.n_stable:
            log.debug("  - Early termination after {} stable iterations.".format(self.n_stable))

        if self.is_convolution:
            X = self.ds.view_converter.topo_view_to_design_mat(X)
        self.ds.X, self.ds.y = X, y

        if self.verbose:
            log.debug("\nEpoch    Validation Error    Time"
                      "\n---------------------------------")

        self._train_layer(self.trainer, self.mlp, self.ds)
        return self

    def _predict(self, X):
        if not self.is_initialized:
            assert self.layers[-1].units is not None,\
                "You must specify the number of units to predict without fitting."
            log.warning("Computing estimates with an untrained network.")
            self._create_specs(X)
            self._create_mlp()

        X, _ = self._reshape(X)
        if X.dtype != numpy.float32:
            X = X.astype(numpy.float32)
        if not isinstance(X, numpy.ndarray):
            X = X.toarray()
        return self.f(X)

    def get_params(self, deep=True):
        result = super(MultiLayerPerceptron, self).get_params(deep=True)
        for l in self.layers:
            result[l.name] = l
        return result


class Regressor(MultiLayerPerceptron, sklearn.base.RegressorMixin):
    """Regressor compatible with sklearn that wraps PyLearn2.
    """

    def fit(self, X, y):
        """Fit the neural network to the given data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_inputs)
            Training vectors as real numbers, where n_samples is the number of
            samples and n_inputs is the number of input features.

        y : array-like, shape (n_samples, n_outputs)
            Target values as real numbers, either as regression targets or
            label probabilities for classification.

        Returns
        -------
        self : object
            Returns this instance.
        """
        return super(Regressor, self)._fit(X, y)

    def predict(self, X):
        """Calculate predictions for specified inputs.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_inputs)
            The input samples as real numbers.

        Returns
        -------
        y : array, shape (n_samples, n_outputs)
            The predicted values as real numbers.
        """
        return super(Regressor, self)._predict(X)


class Classifier(MultiLayerPerceptron, sklearn.base.ClassifierMixin):
    """Classifier compatible with sklearn that wraps PyLearn2.
    """

    def _setup(self):
        super(Classifier, self)._setup()

        # WARNING: Unfortunately, sklearn's LabelBinarizer handles binary data
        # as a special case and encodes it very differently to multiclass cases.
        # In our case, we want to have 2D outputs when there are 2 classes, or
        # the predicted probabilities (e.g. Softmax) will be incorrect.
        # The LabelBinarizer is also implemented in a way that this cannot be
        # customized without a providing a complete rewrite, so here we patch
        # the `type_of_target` function for this to work correctly,
        import sklearn.preprocessing.label as spl
        spl.type_of_target = lambda _: "multiclass"

        self.label_binarizer = sklearn.preprocessing.LabelBinarizer()

    def fit(self, X, y):
        # check now for correct shapes
        assert X.shape[0] == y.shape[0],\
            "Expecting same number of input and output samples."

        # Scan training samples to find all different classes.
        self.label_binarizer.fit(y)
        yp = self.label_binarizer.transform(y)
        # Now train based on a problem transformed into regression.
        return super(Classifier, self)._fit(X, yp, test=y)

    def partial_fit(self, X, y, classes=None):
        if classes is not None:
            self.label_binarizer.fit(classes)
        return self.fit(X, y)

    def predict_proba(self, X):
        """Calculate probability estimates based on these input features.

        Parameters
        ----------
        X : array-like of shape [n_samples, n_features]
            The input data as a numpy array.

        Returns
        -------
        y_prob : array-like of shape [n_samples, n_classes]
            The predicted probability of the sample for each class in the
            model, in the same order as the classes.
        """
        proba = super(Classifier, self)._predict(X)
        return proba / proba.sum(1, keepdims=True)

    def predict(self, X):
        """Predict class by converting the problem to a regression problem.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        y : array-like, shape (n_samples,) or (n_samples, n_classes)
            The predicted classes, or the predicted values.
        """
        y = self.predict_proba(X)
        return self.label_binarizer.inverse_transform(y, threshold=0.5)
