from __future__ import (absolute_import, unicode_literals, print_function)

__all__ = ['MultiLayerPerceptronRegressor', 'MultiLayerPerceptronClassifier']

import os
import time
import logging
import itertools

log = logging.getLogger('sknn')


# By default, we force Theano to use a GPU and fallback to CPU, using 32-bits.
# This must be done in the code before Theano is imported for the first time.
os.environ['THEANO_FLAGS'] = "device=gpu,floatX=float32"

cuda = logging.getLogger('theano.sandbox.cuda')
cuda.setLevel(logging.CRITICAL)
import theano
cuda.setLevel(logging.WARNING)


import numpy
import sklearn.base
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.cross_validation

from pylearn2.datasets import DenseDesignMatrix
from pylearn2.training_algorithms import sgd
from pylearn2.models import mlp, maxout
from pylearn2.costs.mlp.dropout import Dropout
from pylearn2.training_algorithms.learning_rule import RMSProp, Momentum, AdaGrad, AdaDelta
from pylearn2.space import Conv2DSpace
from pylearn2.termination_criteria import MonitorBased


class ansi:
    BOLD = '\033[1;97m'
    WHITE = '\033[0;97m'
    BLUE = '\033[0;94m'
    GREEN = '\033[0;32m'
    ENDC = '\033[0m'


class Layer(object):

    def __init__(self, type, name=None, units=None, pieces=None, channels=None, shape=None, dropout=0.0):
        self.name = name
        self.type = type
        self.units = units
        self.pieces = pieces
        self.channels = channels
        self.shape = shape
        self.dropout = dropout

    def __eq__(self, other):
        return self.__dict__ == other.__dict__



class BaseMLP(sklearn.base.BaseEstimator):
    """
    Abstract base class for wrapping the multi-layer perceptron functionality
    from PyLearn2.

    Parameters
    ----------
    layers : list of tuples
        An iterable sequence of each layer each as a tuple: first with an
        activation type and then optional parameters such as the number of
        units.

            * For hidden layers, you can use the following layer types:
              ``Rectifier``, ``Sigmoid``, ``Tanh``, ``Maxout`` or ``Convolution``.
            * For output layers, you can use the following layer types:
              ``Linear``, ``Softmax`` or ``Gaussian``.

        You must specify at least an output layer, so the last tuple in your
        layers parameter should contain ``Linear`` (for example).

    random_state : int
        Seed for the initialization of the neural network parameters (e.g.
        weights and biases).  This is fully deterministic.

    learning_rule : str
        Name of the learning rule used during stochastic gradient descent,
        one of ``sgd``, ``momentum``, ``nesterov``, ``adadelta`` or ``rmsprop``
        at the moment.

    learning_rate : float
        Real number indicating the default/starting rate of adjustment for
        the weights during gradient descent.  Different learning rules may
        take this into account differently.

    learning_momentum : float
        Real number indicating the momentum factor to be used for the
        learning rule 'momentum'.

    batch_size : int
        Number of training samples to group together when performing stochastic
        gradient descent.  By default each sample is treated on its own.

    n_iter : int
        The number of iterations of gradient descent to perform on the
        neural network's weights when training with ``fit()``.

    valid_set : tuple of array-like
        Validation set (X_v, y_v) to be used explicitly while training.  Both
        arrays should have the same size for the first dimention, and the second
        dimention should match with the training data specified in ``fit()``.

    valid_size : float
        Ratio of the training data to be used for validation.  0.0 means no
        validation, and 1.0 would mean there's no training data!  Common values are
        0.1 or 0.25.

    n_stable : int
        Number of interations after which training should return when the validation
        error remains constant.  This is a sign that the data has been fitted.

    f_stable : float
        Threshold under which the validation error change is assumed to be stable, to
        be used in combination with `n_stable`.

    dropout : bool
        Whether to use drop-out training for the inputs (jittering) and the
        hidden layers, for each training example.

    verbose : bool
        If True, print the score at each epoch via the logger called 'sknn'.  You can
        control the detail of the output by customising the logger level and formatter.
    """

    def __init__(
            self,
            layers,
            random_state=None,
            learning_rule='sgd',
            learning_rate=0.01,
            learning_momentum=0.9,
            dropout=False,
            batch_size=1,
            n_iter=None,
            n_stable=50,
            f_stable=0.001,
            valid_set=None,
            valid_size=0.0,            
            verbose=False):

        self.layers = []
        for i, layer in enumerate(layers):
            if isinstance(layer, tuple):
                if len(layer) == 1:
                    layer = (layer[0], None)
                layer = Layer(layer[0], units=layer[1])

            if layer.name is None:
                label = "Hidden" if i < len(layers)-1 else "Output"
                layer.name = "%s_%i_%s" % (label, i, layer.type)

            assert type(layer) == Layer
            self.layers.append(layer)

        self.random_state = random_state
        self.learning_rule = learning_rule
        self.learning_rate = learning_rate
        self.learning_momentum = learning_momentum
        self.dropout = dropout
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.n_stable = n_stable
        self.f_stable = f_stable
        self.valid_set = valid_set
        self.valid_size = valid_size
        self.verbose = verbose

        self.unit_counts = None
        self.input_space = None
        self.mlp = None
        self.weights = None
        self.vs = None
        self.ds = None
        self.trainer = None
        self.f = None
        self.train_set = None
        self.best_valid_error = float("inf")

        self.cost = "Dropout" if dropout else None
        if learning_rule == 'sgd':
            self._learning_rule = None
        # elif learning_rule == 'adagrad':
        #     self._learning_rule = AdaGrad()
        elif learning_rule == 'adadelta':
            self._learning_rule = AdaDelta()
        elif learning_rule == 'momentum':
            self._learning_rule = Momentum(learning_momentum)
        elif learning_rule == 'nesterov':
            self._learning_rule = Momentum(learning_momentum, nesterov_momentum=True)
        elif learning_rule == 'rmsprop':
            self._learning_rule = RMSProp()
        else:
            raise NotImplementedError(
                "Learning rule type `%s` is not supported." % learning_rule)

        self._setup()

    def _setup(self):
        # raise NotImplementedError("BaseMLP is an abstract class; "
        #                           "use the Classifier or Regressor instead.")
        pass

    def _create_trainer(self, dataset):
        sgd.log.setLevel(logging.WARNING)

        if self.cost == "Dropout":
            self.cost = Dropout(
                input_include_probs={self.layers[0].name: 1.0},
                input_scales={self.layers[0].name: 1.})

        logging.getLogger('pylearn2.monitor').setLevel(logging.WARNING)
        if dataset is not None:
            termination_criterion = MonitorBased(
                channel_name='objective',
                N=self.n_stable,
                prop_decrease=self.f_stable)
        else:
            termination_criterion = None

        return sgd.SGD(
            cost=self.cost,
            batch_size=self.batch_size,
            learning_rule=self._learning_rule,
            learning_rate=self.learning_rate,
            termination_criterion=termination_criterion,
            monitoring_dataset=dataset)

    def _create_hidden_layer(self, name, layer, irange=0.1):
        if layer.type == "Rectifier":
            return mlp.RectifiedLinear(
                layer_name=name,
                dim=layer.units,
                irange=irange)

        if layer.type == "Sigmoid":
            return mlp.Sigmoid(
                layer_name=name,
                dim=layer.units,
                irange=irange)

        if layer.type == "Tanh":
            return mlp.Tanh(
                layer_name=name,
                dim=layer.units,
                irange=irange)

        if layer.type == "Maxout":
            return maxout.Maxout(
                layer_name=name,
                num_units=layer.units,
                num_pieces=layer.pieces,
                irange=irange)

        if layer.type == "Convolution":
            return mlp.ConvRectifiedLinear(
                layer_name=name,
                output_channels=layer.channels,
                kernel_shape=layer.shape,
                pool_shape=(1,1),
                pool_stride=(1,1),
                irange=irange)

        raise NotImplementedError(
            "Hidden layer type `%s` is not implemented." % layer.type)

    def _create_output_layer(self, layer):
        fan_in = self.unit_counts[-2]
        fan_out = self.unit_counts[-1]
        lim = numpy.sqrt(6) / (numpy.sqrt(fan_in + fan_out))

        if layer.type == "Linear":
            return mlp.Linear(
                dim=layer.units,
                layer_name=layer.name,
                irange=lim)

        if layer.type == "Gaussian":
            return mlp.LinearGaussian(
                init_beta=0.1,
                min_beta=0.001,
                max_beta=1000,
                beta_lr_scale=None,
                dim=layer.units,
                layer_name=layer.name,
                irange=lim)

        if layer.type == "Softmax":
            return mlp.Softmax(
                layer_name=layer.name,
                n_classes=layer.units,
                irange=lim)

        raise NotImplementedError(
            "Output layer type `%s` is not implemented." % layer.type)

    def _create_mlp(self):
        # Create the layers one by one, connecting to previous.
        mlp_layers = []
        for i, layer in enumerate(self.layers[:-1]):
            fan_in = self.unit_counts[i]
            fan_out = self.unit_counts[i + 1]

            lim = numpy.sqrt(6) / numpy.sqrt(fan_in + fan_out)
            if layer.type == "Tanh":
                lim *= 1.1 * lim
            elif layer.type in ("Rectifier", "Maxout", "Convolution"):
                # He, Rang, Zhen and Sun, converted to uniform.
                lim *= numpy.sqrt(2)
            elif layer.type == "Sigmoid":
                lim *= 4

            hidden_layer = self._create_hidden_layer(layer.name, layer, irange=lim)
            mlp_layers.append(hidden_layer)

        # Deal with output layer as a special case.
        output_layer = self._create_output_layer(self.layers[-1])
        mlp_layers.append(output_layer)

        self.mlp = mlp.MLP(
            mlp_layers,
            nvis=None if self.is_convolution else self.unit_counts[0],
            seed=self.random_state,
            input_space=self.input_space)

        if self.weights is not None:
            self._array_to_mlp(self.weights, self.mlp)
            self.weights = None

        inputs = self.mlp.get_input_space().make_theano_batch()
        self.f = theano.function([inputs], self.mlp.fprop(inputs))

    def _create_matrix_input(self, X, y):
        if self.is_convolution:
            # b01c arrangement of data
            # http://benanne.github.io/2014/04/03/faster-convolutions-in-theano.html for more
            # input: (batch size, channels, rows, columns)
            # filters: (number of filters, channels, rows, columns)
            input_space = Conv2DSpace(shape=X.shape[1:3], num_channels=X.shape[-1])
            view = input_space.get_origin_batch(X.shape[0])
            return DenseDesignMatrix(topo_view=view, y=y), input_space
        else:
            return DenseDesignMatrix(X=X, y=y), None

    def _initialize(self, X, y):
        assert not self.is_initialized,\
            "This neural network has already been initialized."

        log.info(
            "Initializing neural network with %i layers, %i inputs and %i outputs.",
            len(self.layers), X.shape[1], y.shape[1])

        # Calculate and store all layer sizes.
        if self.layers[-1].units is None:
            self.layers[-1].units = y.shape[1]
        else:
            assert self.layers[-1].units == y.shape[1],\
                "Mismatch between dataset size and units in output layer."

        self.unit_counts = [X.shape[1]]
        for layer in self.layers:
            if layer.units is not None:
                self.unit_counts.append(layer.units)
            else:
                # TODO: Compute correct number of outputs for convolution.
                self.unit_counts.append(layer.channels)

            log.debug("  - Type: {}{: <10}{}  Units: {}{: <4}{}".format(
                ansi.BOLD, layer.type, ansi.ENDC, ansi.BOLD, layer.units, ansi.ENDC))
        log.debug("")

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

        self.trainer = self._create_trainer(self.vs)
        self.trainer.setup(self.mlp, self.ds)
        

    @property
    def is_initialized(self):
        """Check if the neural network was setup already.
        """
        return not (self.mlp is None or self.f is None)

    @property
    def is_convolution(self):
        """Check whether this neural network includes convolution layers.
        """
        return "Conv" in self.layers[0].type

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

    def _fit(self, X, y, test=None):
        assert X.shape[0] == y.shape[0],\
            "Expecting same number of input and output samples."
        num_samples, data_size = X.shape[0], X.size+y.size

        if y.ndim == 1:
            y = y.reshape((y.shape[0], 1))
        if not isinstance(X, numpy.ndarray):
            X = X.toarray()
        if not isinstance(y, numpy.ndarray):
            y = y.toarray()

        if not self.is_initialized:            
            self._initialize(X, y)
            X, y = self.train_set
        else:
            self.train_set = X, y

        if self.is_convolution:
            X = self.ds.view_converter.topo_view_to_design_mat(X)
        self.ds.X, self.ds.y = X, y

        # Bug in PyLearn2 that has some unicode channels, can't sort.
        self.mlp.monitor.channels = {str(k): v for k, v in self.mlp.monitor.channels.items()}

        log.info("Training on dataset of {:,} samples with {:,} total size.".format(num_samples, data_size))
        if self.valid_set:
            X_v, _ = self.valid_set
            log.debug("  - Train: {: <9,}  Valid: {: <4,}".format(X.shape[0], X_v.shape[0]))
        if self.n_iter:
            log.debug("  - Terminating loop after {} total iterations.".format(self.n_iter))
        if self.n_stable:
            log.debug("  - Early termination after {} stable iterations.".format(self.n_stable))

        log.debug("""
Epoch    Validation Error    Time
---------------------------------""")

        for i in itertools.count(0):
            start = time.time()
            self.trainer.train(dataset=self.ds)

            self.mlp.monitor.report_epoch()
            self.mlp.monitor()

            if not self.trainer.continue_learning(self.mlp):
                log.debug("")
                log.info("Early termination condition fired at %i iterations.", i)
                break
            if self.n_iter is not None and i >= self.n_iter:
                log.debug("")
                log.info("Terminating after specified %i total iterations.", i)
                break

            if self.verbose:
                objective = self.mlp.monitor.channels.get('objective', None)
                if objective:
                    avg_valid_error = objective.val_shared.get_value()
                    self.best_valid_error = min(self.best_valid_error, avg_valid_error)
                else:
                    avg_valid_error = None

                best_valid = bool(self.best_valid_error == avg_valid_error)
                log.debug("{:>5}      {}{}{}        {:>3.1f}s".format(
                          i,
                          ansi.GREEN if best_valid else "",
                          "{:>10.6f}".format(float(avg_valid_error)) if avg_valid_error else "     N/A  ",
                          ansi.ENDC if best_valid else "",
                          time.time() - start
                          ))

        return self

    def _predict(self, X):
        if not self.is_initialized:
            raise ValueError("The neural network has not been trained.")

        if X.dtype != numpy.float32:
            X = X.astype(numpy.float32)
        if not isinstance(X, numpy.ndarray):
            X = X.toarray()

        return self.f(X)



class MultiLayerPerceptronRegressor(BaseMLP, sklearn.base.RegressorMixin):
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
        return super(MultiLayerPerceptronRegressor, self)._fit(X, y)

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
        return super(MultiLayerPerceptronRegressor, self)._predict(X)



class MultiLayerPerceptronClassifier(BaseMLP, sklearn.base.ClassifierMixin):
    """Classifier compatible with sklearn that wraps PyLearn2.
    """

    def _setup(self):
        # WARNING: Unfortunately, sklearn's LabelBinarizer handles binary data
        # as a special case and encodes it very differently to multiclass cases.
        # In our case, we want to have 2D outputs when there are 2 classes, or
        # the predicted probabilities (e.g. Softmax) will be incorrect.
        # The LabelBinarizer is also implemented in a way that this cannot be
        # customized without a providing a complete rewrite, so here we patch
        # the `type_of_target` function for this to work correctly,
        import sklearn.preprocessing.label as L
        L.type_of_target = lambda _: "multiclass"

        self.label_binarizer = sklearn.preprocessing.LabelBinarizer()

    def fit(self, X, y):
        # check now for correct shapes
        assert X.shape[0] == y.shape[0],\
            "Expecting same number of input and output samples."

        # Scan training samples to find all different classes.
        self.label_binarizer.fit(y)
        yp = self.label_binarizer.transform(y)
        # Now train based on a problem transformed into regression.
        return super(MultiLayerPerceptronClassifier, self)._fit(X, yp, test=y)

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
        proba = super(MultiLayerPerceptronClassifier, self)._predict(X)

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
