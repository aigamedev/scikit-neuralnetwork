# -*- coding: utf-8 -*-
from __future__ import (absolute_import, unicode_literals, print_function)

__all__ = ['Regressor', 'Classifier', 'Layer', 'Convolution']

import os
import sys
import math
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

from .nn import NeuralNetwork, Layer, Convolution, ansi

from .backend import MultiLayerPerceptronBackend


class MultiLayerPerceptron(NeuralNetwork, sklearn.base.BaseEstimator):
    # Abstract base class for wrapping multi-layer perceptron functionality.

    def _setup(self):
        pass

    def _initialize(self, X, y=None):
        assert not self.is_initialized,\
            "This neural network has already been initialized."
        self._create_specs(X, y)

        self._backend = MultiLayerPerceptronBackend(self)
        return self._backend._initialize_impl(X, y)

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
                assert l.kernel_shape is not None,\
                    "Layer `%s` requires parameter `kernel_shape` to be set." % (l.name,)
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

    def __getstate__(self):
        d = self.__dict__.copy()

        # If the MLP does not exist, then the client code is trying to serialize
        # this object to communicate between multiple processes.
        if self._backend is not None:
            d['weights'] = self._backend._mlp_to_array()

        for k in [k for k in d.keys() if k.startswith('_')]:
            del d[k]
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

        # Only create the MLP if the weights were serialized. Otherwise, it
        # may have been serialized for multiprocessing reasons pre-training.
        self._create_logger()
        self._backend = None

    def _reshape(self, X, y=None):
        if y is not None and y.ndim == 1:
            y = y.reshape((y.shape[0], 1))
        if self.is_convolution and X.ndim == 3:
            X = X.reshape((X.shape[0], X.shape[1], X.shape[2], 1))
        if self.is_convolution and X.ndim == 2:
            size = math.sqrt(X.shape[1])
            assert size.is_integer(),\
                "Input array is not in image shape, and could not assume a square."
            X = X.reshape((X.shape[0], int(size), int(size), 1))
        if not self.is_convolution and X.ndim > 2:
            X = X.reshape((X.shape[0], numpy.product(X.shape[1:])))
        return X, y

    def _fit(self, X, y):
        assert X.shape[0] == y.shape[0],\
            "Expecting same number of input and output samples."
        data_shape, data_size = X.shape, X.size+y.size
        X, y = self._reshape(X, y)

        if not self.is_initialized:
            X, y = self._initialize(X, y)

        log.info("Training on dataset of {:,} samples with {:,} total size.".format(data_shape[0], data_size))
        if data_shape[1:] != X.shape[1:]:
            log.warning("  - Reshaping input array from {} to {}.".format(data_shape, X.shape))
        if self.valid_set is not None:
            X_v, _ = self.valid_set
            log.debug("  - Train: {: <9,}  Valid: {: <4,}".format(X.shape[0], X_v.shape[0]))
        if self.regularize is not None:
            comment = ", auto-enabled from layers" if self.regularize is None else ""
            log.debug("  - Using `%s` for regularization%s." % (self.regularize, comment))
        if self.n_iter is not None:
            log.debug("  - Terminating loop after {} total iterations.".format(self.n_iter))
        if self.n_stable is not None and self.n_stable < (self.n_iter or sys.maxsize):
            log.debug("  - Early termination after {} stable iterations.".format(self.n_stable))

        if self.verbose:
            log.debug("\nEpoch    Validation Error      Time"
                      "\n-----------------------------------")

        try:
            self._backend._train_impl(X, y)
        except RuntimeError as e:
            log.error("\n{}{}{}\n\n{}\n".format(
                ansi.RED,
                "A runtime exception was caught during training. This likely occurred due to\n"
                "a divergence of the SGD algorithm, and NaN floats were found by PyLearn2.",
                ansi.ENDC,
                "Try setting the `learning_rate` 10x lower to resolve this, for example:\n"
                "    learning_rate=%f" % (self.learning_rate * 0.1)))
            raise e

        return self

    def _predict(self, X):
        X, _ = self._reshape(X)

        if self._backend is None:
            assert self.layers[-1].units is not None,\
                "You must specify the number of units to predict without fitting."
            if self.weights is None:
                log.warning("WARNING: Computing estimates with an untrained network.")
            self._initialize(X)

        if not isinstance(X, numpy.ndarray):
            X = X.toarray()
        return self._backend._predict_impl(X)

    def get_params(self, deep=True):
        result = super(MultiLayerPerceptron, self).get_params(deep=True)
        for l in self.layers:
            result[l.name] = l
        return result


class Regressor(MultiLayerPerceptron, sklearn.base.RegressorMixin):
    # Regressor compatible with sklearn that wraps various NN implementations.

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
    # Classifier compatible with sklearn that wraps various NN implementations.

    def _setup(self):
        super(Classifier, self)._setup()
        self.label_binarizers = []

        # WARNING: Unfortunately, sklearn's LabelBinarizer handles binary data
        # as a special case and encodes it very differently to multiclass cases.
        # In our case, we want to have 2D outputs when there are 2 classes, or
        # the predicted probabilities (e.g. Softmax) will be incorrect.
        # The LabelBinarizer is also implemented in a way that this cannot be
        # customized without a providing a complete rewrite, so here we patch
        # the `type_of_target` function for this to work correctly,
        import sklearn.preprocessing.label as spl
        spl.type_of_target = lambda _: "multiclass"

    def fit(self, X, y):
        # check now for correct shapes
        assert X.shape[0] == y.shape[0],\
            "Expecting same number of input and output samples."
        if y.ndim == 1:
            y = y.reshape((y.shape[0], 1))

        if y.shape[1] == 1 and self.layers[-1].type != 'Softmax':
            log.warning('{}WARNING: Expecting `Softmax` type for the last layer '
                        'in classifier.{}\n'.format(ansi.YELLOW, ansi.ENDC))
        if y.shape[1] > 1 and self.layers[-1].type != 'Sigmoid':
            log.warning('{}WARNING: Expecting `Sigmoid` for last layer in '
                        'multi-output classifier.{}\n'.format(ansi.YELLOW, ansi.ENDC))

        # Deal deal with single- and multi-output classification problems.
        self.label_binarizers = [sklearn.preprocessing.LabelBinarizer() for _ in range(y.shape[1])]
        ys = [lb.fit_transform(y[:,i]) for i, lb in enumerate(self.label_binarizers)]
        yp = numpy.concatenate(ys, axis=1)

        # Also transform the validation set if it was explicitly specified.
        if self.valid_set is not None:
            X_v, y_v = self.valid_set
            if y_v.ndim == 1:
                y_v = y_v.reshape((y_v.shape[0], 1))
            ys = [lb.transform(y_v[:,i]) for i, lb in enumerate(self.label_binarizers)]
            y_vp = numpy.concatenate(ys, axis=1)
            self.valid_set = self._reshape(X_v, y_vp)
 
        # Now train based on a problem transformed into regression.
        return super(Classifier, self)._fit(X, yp)

    def partial_fit(self, X, y, classes=None):
        if y.ndim == 1:
            y = y.reshape((y.shape[0], 1))

        if classes is not None:
            if isinstance(classes[0], int):
                classes = [classes]
            self.label_binarizers = [sklearn.preprocessing.LabelBinarizer() for _ in range(y.shape[1])]
            for lb, cls in zip(self.label_binarizers, classes):
                lb.fit(cls)
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
        index = 0
        for lb in self.label_binarizers:
            sz = len(lb.classes_)
            proba[:,index:index+sz] /= proba[:,index:index+sz].sum(1, keepdims=True) 
            index += sz
        return proba

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
        assert self.label_binarizers != [],\
            "Can't predict without fitting: output classes are unknown."

        yp = self.predict_proba(X)
        ys = []
        index = 0
        for lb in self.label_binarizers:
            sz = len(lb.classes_)
            y = lb.inverse_transform(yp[:,index:index+sz], threshold=0.5)
            ys.append(y.reshape((y.shape[0], 1)))
            index += sz
        y = numpy.concatenate(ys, axis=1)
        return y
