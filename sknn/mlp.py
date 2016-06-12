# -*- coding: utf-8 -*-
from __future__ import (absolute_import, unicode_literals, print_function)

__all__ = ['Regressor', 'Classifier', 'Layer', 'Convolution']

import os
import sys
import math
import time
import logging
import itertools
import contextlib

log = logging.getLogger('sknn')


import numpy
import theano
import sklearn.base
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.cross_validation

from .nn import NeuralNetwork, Layer, Convolution, Native, ansi
from . import backend


class MultiLayerPerceptron(NeuralNetwork, sklearn.base.BaseEstimator):
    # Abstract base class for wrapping multi-layer perceptron functionality.
    __doc__ = NeuralNetwork.__doc__

    def _setup(self):
        pass

    def _initialize(self, X, y=None, w=None):
        assert not self.is_initialized,\
            "This neural network has already been initialized."
        self._create_specs(X, y)

        backend.setup()
        self._backend = backend.MultiLayerPerceptronBackend(self)
        return self._backend._initialize_impl(X, y, w)

    def _check_layer(self, layer, required, optional=[]):
        required.extend(['name', 'type'])
        for r in required:
            if getattr(layer, r) is None: raise\
                ValueError("Layer type `%s` requires parameter `%s`."\
                           % (layer.type, r))

        optional.extend(['weight_decay', 'dropout', 'normalize', 'frozen'])
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
        self.unit_counts = [numpy.product(X.shape[1:]) if self.is_convolution() else X.shape[1]]
        res = X.shape[1:3] if self.is_convolution() else None

        for l in self.layers:
            if isinstance(l, Convolution):
                assert l.kernel_shape is not None,\
                    "Layer `%s` requires parameter `kernel_shape` to be set." % (l.name,)
                if l.border_mode == 'valid':
                    res = (int((res[0] - l.kernel_shape[0]) / l.pool_shape[0]) + 1,
                           int((res[1] - l.kernel_shape[1]) / l.pool_shape[1]) + 1)
                if l.border_mode == 'full':
                    res = (int((res[0] + l.kernel_shape[0]) / l.pool_shape[0]) - 1,
                           int((res[1] + l.kernel_shape[1]) / l.pool_shape[1]) - 1)
                           
                if l.scale_factor != (1, 1):
                    res = (int(l.scale_factor[0] * res[0]), int(l.scale_factor[1] * res[1]))
 
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
        d['valid_set'] = None

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
        if self.is_convolution() and X.ndim == 3:
            X = X.reshape((X.shape[0], X.shape[1], X.shape[2], 1))
        if self.is_convolution() and X.ndim == 2:
            size = math.sqrt(X.shape[1])
            assert size.is_integer(),\
                "Input array is not in image shape, and could not assume a square."
            X = X.reshape((X.shape[0], int(size), int(size), 1))
        if not self.is_convolution() and X.ndim > 2:
            X = X.reshape((X.shape[0], numpy.product(X.shape[1:])))
        return X, y

    def _do_callback(self, event, variables):
        if self.callback is None:
            return

        del variables['self']
        if isinstance(self.callback, dict):
            function = self.callback.get(event, None)
            return function(**variables) if function else True
        else:
            return self.callback(event, **variables)

    def _train(self, X, y, w=None):
        assert self.n_iter or self.n_stable,\
            "Neither n_iter nor n_stable were specified; training would loop forever."

        best_train_error, best_valid_error = float("inf"), float("inf")
        best_params = [] 
        n_stable = 0
        self._do_callback('on_train_start', locals())

        for i in itertools.count(1):
            start_time = time.time()
            self._do_callback('on_epoch_start', locals())

            is_best_train = False
            avg_train_error = self._backend._train_impl(X, y, w)
            if avg_train_error is not None:
                if math.isnan(avg_train_error):
                    raise RuntimeError("Training diverged and returned NaN.")
                
                best_train_error = min(best_train_error, avg_train_error)
                is_best_train = bool(avg_train_error < best_train_error * (1.0 + self.f_stable))

            is_best_valid = False
            avg_valid_error = None
            if self.valid_set is not None:
                avg_valid_error = self._backend._valid_impl(*self.valid_set)
                if avg_valid_error is not None:
                    best_valid_error = min(best_valid_error, avg_valid_error)
                    is_best_valid = bool(avg_valid_error < best_valid_error * (1.0 + self.f_stable))

            finish_time = time.time()
            log.debug("\r{:>5}         {}{}{}            {}{}{}        {:>5.1f}s".format(
                      i,
                      ansi.BLUE if is_best_train else "",
                      "{0:>10.3e}".format(float(avg_train_error)) if (avg_train_error is not None) else "     N/A  ",
                      ansi.ENDC if is_best_train else "",

                      ansi.GREEN if is_best_valid else "",
                      "{:>10.3e}".format(float(avg_valid_error)) if (avg_valid_error is not None) else "     N/A  ",
                      ansi.ENDC if is_best_valid else "",

                      finish_time - start_time
                      ))

            if is_best_valid or (self.valid_set is None and is_best_train):
                best_params = self._backend._mlp_to_array()
                n_stable = 0
            else:
                n_stable += 1

            if self._do_callback('on_epoch_finish', locals()) == False:
                log.debug("")
                log.info("User defined callback terminated at %i iterations.", i)
                break

            if self.n_stable is not None and n_stable >= self.n_stable:
                log.debug("")
                log.info("Early termination condition fired at %i iterations.", i)
                break
            if self.n_iter is not None and i >= self.n_iter:
                log.debug("")
                log.info("Terminating after specified %i total iterations.", i)
                break

        self._do_callback('on_train_finish', locals())
        self._backend._array_to_mlp(best_params, self._backend.mlp)

    def _fit(self, X, y, w=None):
        assert X.shape[0] == y.shape[0],\
            "Expecting same number of input and output samples."
        data_shape = X.shape
        known_size = hasattr(X, 'size') and hasattr(y, 'size')
        data_size = '{:,}'.format(X.size+y.size) if known_size else 'N/A'
        X, y = self._reshape(X, y)

        if not self.is_initialized:
            X, y = self._initialize(X, y, w)

        log.info("Training on dataset of {:,} samples with {} total size.".format(data_shape[0], data_size))
        if data_shape[1:] != X.shape[1:]:
            log.warning("  - Reshaping input array from {} to {}.".format(data_shape, X.shape))
        if self.valid_set is not None:
            X_v, _ = self.valid_set
            log.debug("  - Train: {: <9,}  Valid: {: <4,}".format(X.shape[0], X_v.shape[0]))
        regularize = self.regularize or self.auto_enabled.get('regularize', None)
        if regularize is not None:
            comment = ", auto-enabled from layers" if 'regularize' in self.auto_enabled else "" 
            log.debug("  - Using `%s` for regularization%s." % (regularize, comment))
        normalize = self.normalize or self.auto_enabled.get('normalize', None)
        if normalize is not None:
            comment = ", auto-enabled from layers" if 'normalize' in self.auto_enabled else ""
            log.debug("  - Using `%s` normalization%s." % (normalize, comment))
        if self.n_iter is not None:
            log.debug("  - Terminating loop after {} total iterations.".format(self.n_iter))
        if self.n_stable is not None and self.n_stable < (self.n_iter or sys.maxsize):
            log.debug("  - Early termination after {} stable iterations.".format(self.n_stable))

        if self.verbose:
            log.debug("\nEpoch       Training Error       Validation Error       Time"
                      "\n------------------------------------------------------------")

        try:
            self._train(X, y, w)
        except RuntimeError as e:
            log.error("\n{}{}{}\n\n{}\n".format(
                ansi.RED,
                "A runtime exception was caught during training. This likely occurred due to\n"
                "a divergence of the SGD algorithm, and NaN floats were found by the backend.",
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

        return self._backend._predict_impl(X)

    def get_params(self, deep=True):
        result = super(MultiLayerPerceptron, self).get_params(deep=True)
        for l in self.layers:
            result[l.name] = l
        return result


class Regressor(MultiLayerPerceptron, sklearn.base.RegressorMixin):
    # Regressor compatible with sklearn that wraps various NN implementations.
    # The constructor and bulk of documentation is inherited from MultiLayerPerceptron.

    def fit(self, X, y, w=None):
        """Fit the neural network to the given continuous data as a regression problem.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_inputs)
            Training vectors as real numbers, where n_samples is the number of
            samples and n_inputs is the number of input features.

        y : array-like, shape (n_samples, n_outputs)
            Target values are real numbers used as regression targets.

        w : array-like (optional), shape (n_samples) 
            Floating point weights for each of the training samples, used as mask to
            modify the cost function during optimization. 

        Returns
        -------
        self : object
            Returns this instance.
        """

        if self.valid_set is not None:
            self.valid_set = self._reshape(*self.valid_set)

        return super(Regressor, self)._fit(X, y, w)

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

    @property
    def is_classifier(self):
        return False


class Classifier(MultiLayerPerceptron, sklearn.base.ClassifierMixin):
    # Classifier compatible with sklearn that wraps various NN implementations.

    def _setup(self):
        super(Classifier, self)._setup()
        self.label_binarizers = []

    @contextlib.contextmanager
    def _patch_sklearn(self):
        # WARNING: Unfortunately, sklearn's LabelBinarizer handles binary data
        # as a special case and encodes it very differently to multiclass cases.
        # In our case, we want to have 2D outputs when there are 2 classes, or
        # the predicted probabilities (e.g. Softmax) will be incorrect.
        # The LabelBinarizer is also implemented in a way that this cannot be
        # customized without a providing a near-complete rewrite, so here we patch
        # the `type_of_target` function for this to work correctly.
        import sklearn.preprocessing.label as spl
        backup = spl.type_of_target 
        spl.type_of_target = lambda _: "multiclass"
        yield
        spl.type_of_target = backup

    def fit(self, X, y, w=None):
        """Fit the neural network to symbolic labels as a classification problem.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vectors as real numbers, where n_samples is the number of
            samples and n_inputs is the number of input features.

        y : array-like, shape (n_samples, n_classes)
            Target values as integer symbols, for either single- or multi-output
            classification problems.

        w : array-like (optional), shape (n_samples) 
            Floating point weights for each of the training samples, used as mask to
            modify the cost function during optimization.

        Returns
        -------
        self : object
            Returns this instance.
        """

        assert X.shape[0] == y.shape[0],\
            "Expecting same number of input and output samples."
        if y.ndim == 1:
            y = y.reshape((y.shape[0], 1))

        if y.shape[1] == 1 and self.layers[-1].type != 'Softmax':
            log.warning('{}WARNING: Expecting `Softmax` type for the last layer '
                        'in classifier.{}\n'.format(ansi.YELLOW, ansi.ENDC))
        if y.shape[1] > 1 and self.layers[-1].type != 'Sigmoid':
            log.warning('{}WARNING: Expecting `Sigmoid` as last layer in '
                        'multi-output classifier.{}\n'.format(ansi.YELLOW, ansi.ENDC))

        # Deal deal with single- and multi-output classification problems.
        LB = sklearn.preprocessing.LabelBinarizer
        self.label_binarizers = [LB() for _ in range(y.shape[1])]
        with self._patch_sklearn():
            ys = [lb.fit_transform(y[:,i]) for i, lb in enumerate(self.label_binarizers)]
        yp = numpy.concatenate(ys, axis=1).astype(theano.config.floatX)

        # Also transform the validation set if it was explicitly specified.
        if self.valid_set is not None:
            X_v, y_v = self.valid_set
            if y_v.ndim == 1:
                y_v = y_v.reshape((y_v.shape[0], 1))
            with self._patch_sklearn():
                ys = [lb.transform(y_v[:,i]) for i, lb in enumerate(self.label_binarizers)]
            y_vp = numpy.concatenate(ys, axis=1)
            self.valid_set = (X_v, y_vp)

        # Now train based on a problem transformed into regression.
        return super(Classifier, self)._fit(X, yp, w)

    def partial_fit(self, X, y, classes=None):
        if y.ndim == 1:
            y = y.reshape((y.shape[0], 1))

        if classes is not None:
            if isinstance(classes[0], int):
                classes = [classes]
            LB = sklearn.preprocessing.LabelBinarizer
            self.label_binarizers = [LB() for _ in range(y.shape[1])]
            for lb, cls in zip(self.label_binarizers, classes):
                lb.fit(cls)

        return self.fit(X, y)

    def predict_proba(self, X, collapse=True):
        """Calculate probability estimates based on these input features.

        Parameters
        ----------
        X : array-like of shape [n_samples, n_features]
            The input data as a numpy array.

        Returns
        -------
        y_prob : list of arrays of shape [n_samples, n_features, n_classes]
            The predicted probability of the sample for each class in the
            model, in the same order as the classes.
        """
        proba = super(Classifier, self)._predict(X)
        index, yp = 0, []
        for lb in self.label_binarizers:
            sz = len(lb.classes_)
            p = proba[:,index:index+sz]
            yp.append(p / p.sum(1, keepdims=True))
            index += sz
        return yp[0] if (len(yp) == 1 and collapse) else yp

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

        yp = self.predict_proba(X, collapse=False)
        ys = []
        index = 0
        for lb, p in zip(self.label_binarizers, yp):
             sz = len(lb.classes_)
             y = lb.inverse_transform(p, threshold=0.5)
             ys.append(y.reshape((-1, 1)))
             index += sz
        return numpy.concatenate(ys, axis=1)

    @property
    def is_classifier(self):
        return True
    
    @property
    def classes_(self):
        """Return a list of class labels used for each feature. For single feature
        classification, the index of the label in the array is the same as returned
        by `predict_proba()` (e.g. labels `[-1, 0, +1]` mean indices `[0, 1, 2]`).
        
        In the case of multiple feature classification, the index of the label must
        be offset by the number of labels for previous features.   For example, 
        if the second feature also has labels `[-1, 0, +1]` its indicies will be
        `[3, 4, 5]` resuming from the first feature in the array returned by
        `predict_proba()`. 
        
        Returns
        -------
        c : list of array, shape (n_classes, n_labels)
            List of the labels as integers used for each feature.
        """ 
        assert self.label_binarizers != [],\
            "There are no output classes because classifier isn't fitted."
        
        return [lb.classes_ for lb in self.label_binarizers]
