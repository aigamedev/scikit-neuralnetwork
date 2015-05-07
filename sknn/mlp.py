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

from .pywrap2 import (datasets, space, sgd, mlp, maxout, dropout)
from .pywrap2 import learning_rule as lr, termination_criteria as tc

from .dataset import SparseDesignMatrix


class ansi:
    BOLD = '\033[1;97m'
    WHITE = '\033[0;97m'
    BLUE = '\033[0;94m'
    GREEN = '\033[0;32m'
    ENDC = '\033[0m'


class Layer(object):
    """
    Specification for a layer to be passed to the neural network during construction.  This
    includes a variety of parameters to configure each layer based on its activation type.

    Parameters
    ----------

    type: str
        Select which activation function this layer should use, as a string.  Specifically,
        options are ``Rectifier``, ``Sigmoid``, ``Tanh``, and ``Maxout`` for non-linear layers
        and ``Linear``, ``Softmax`` or ``Gaussian`` for linear layers.

    name: str, optional
        You optionally can specify a name for this layer, and its parameters
        will then be accessible to `scikit-learn` via a nested sub-object.  For example,
        if name is set to `hidden1`, then the parameter `hidden1__units` from the network
        is bound to this layer's `units` variable.

    units: int
        The number of units (also known as neurons) in this layer.  This applies to all
        layer types except for convolution.

    pieces: int, optional
        The number of piecewise linear segments in the Maxout activation.  This is
        optional and only applies when `Maxout` is selected as the layer type.

    dropout: float, optional
        The ratio of inputs to drop out for this layer during training.  For example, 0.25
        means that 25% of the inputs will be excluded for each training sample, with the
        remaining inputs being renormalized accordingly.

    warning: None
        You should use keyword arguments after `type` when initializing this object. If not,
        the code will raise an AssertionError.
    """

    def __init__(
            self,
            type,
            warning=None,
            name=None,
            units=None,
            pieces=None,
            dropout=None):

        assert warning is None,\
            "Specify layer parameters as keyword arguments, not positional arguments."

        if type not in ['Rectifier', 'Sigmoid', 'Tanh', 'Maxout',
                        'Linear', 'Softmax', 'Gaussian']:
            raise NotImplementedError("Layer type `%s` is not implemented." % type)

        self.name = name
        self.type = type
        self.units = units
        self.pieces = pieces
        self.dropout = dropout

    def set_params(self, **params):
        """Setter for internal variables that's compatible with ``scikit-learn``.
        """
        for k, v in params.items():
            if k not in self.__dict__:
                raise ValueError("Invalid parameter `%s` for layer `%s`." % (k, self.name))
            self.__dict__[k] = v

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __repr__(self):
        copy = self.__dict__.copy()
        del copy['type']
        params = ", ".join(["%s=%r" % (k, v) for k, v in copy.items() if v is not None])
        return "<sknn.mlp.%s `%s`: %s>" % (self.__class__.__name__, self.type, params)


class Convolution(Layer):
    """
    Specification for a convolution layer to be passed to the neural network in construction.
    This includes a variety of convolution-specific parameters to configure each layer, as well
    as activation-specific parameters.

    Parameters
    ----------

    type: str
        Select which activation function this convolution layer should use, as a string.
        For hidden layers, you can use the following convolution types ``Rectifier``,
        ``Sigmoid``, ``Tanh`` or ``Linear``.

    name: str, optional
        You optionally can specify a name for this layer, and its parameters
        will then be accessible to `scikit-learn` via a nested sub-object.  For example,
        if name is set to `hidden1`, then the parameter `hidden1__units` from the network
        is bound to this layer's `units` variable.

    pieces: int, optional
        The number of piecewise linear segments in the Maxout activation.  This is
        optional and only applies when `Maxout` is selected as the layer type.

    channels: int
        Number of output channels for the convolution layers.  Each channel has its own
        set of shared weights which are trained by applying the kernel over the image.

    kernel_shape: tuple of ints
        A two-dimensional tuple of integers corresponding to the shape of the kernel when
        convolution is used.  For example, this could be a square kernel `(3,3)` or a full
        horizontal or vertical kernel on the input matrix, e.g. `(N,1)` or `(1,N)`.

    kernel_stride: tuple of ints, optional
        A two-dimensional tuple of integers that represents the steps taken by the kernel
        through the input image.  By default, this is set to the same as `pool_shape` but can
        be customized separately even if pooling is turned off.

    border_mode: str
        String indicating the way borders in the image should be processed, one of two options:

            * `valid` — Only pixels from input where the kernel fits within bounds are processed.
            * `full` — All pixels from input are processed, and the boundaries are zero-padded.

        The size of the output will depend on this mode, for `full` it's identical to the input,
        but for `valid` it will be smaller or equal.

    pool_shape: tuple of ints, optional
        A two-dimensional tuple of integers corresponding to the pool size.  This should be
        square, for example `(2,2)` to reduce the size by half, or `(4,4)` to make the output
        a quarter of the original.

    pool_type: str, optional
        Type of the pooling to be used; can be either `max` or `mean`.  If a `pool_shape` is
        specified the default is to take the maximum value of all inputs that fall into this
        pool. Otherwise, the default is None and no pooling is used for performance.

    dropout: float, optional
        The ratio of inputs to drop out for this layer during training.  For example, 0.25
        means that 25% of the inputs will be excluded for each training sample, with the
        remaining inputs being renormalized accordingly.

    warning: None
        You should use keyword arguments after `type` when initializing this object. If not,
        the code will raise an AssertionError.
    """
    def __init__(
            self,
            type,
            warning=None,
            name=None,
            channels=None,
            pieces=None,
            kernel_shape=None,
            kernel_stride=None,
            border_mode='valid',
            pool_shape=None,
            pool_type=None,
            dropout=None):

        assert warning is None,\
            "Specify layer parameters as keyword arguments, not positional arguments."

        if type not in ['Rectifier', 'Sigmoid', 'Tanh', 'Linear']:
            raise NotImplementedError("Convolution type `%s` is not implemented." % (type,))
        if border_mode not in ['valid', 'full']:
            raise NotImplementedError("Convolution border_mode `%s` is not implemented." % (border_mode,))

        super(Convolution, self).__init__(
                type,
                name=name,
                pieces=pieces,
                dropout=dropout)

        self.channels = channels
        self.pool_shape = pool_shape or (1,1)
        self.pool_type = pool_type or ('max' if pool_shape else None)
        self.kernel_shape = kernel_shape
        self.kernel_stride = kernel_stride or self.pool_shape
        self.border_mode = border_mode


class MultiLayerPerceptron(sklearn.base.BaseEstimator):
    """
    Abstract base class for wrapping the multi-layer perceptron functionality
    from PyLearn2.

    Parameters
    ----------

    layers: list of Layer
        An iterable sequence of each layer each as a :class:`sknn.mlp.Layer` instance that
        contains its type, optional name, and any paramaters required.

            * For hidden layers, you can use the following layer types:
              ``Rectifier``, ``Sigmoid``, ``Tanh``, ``Maxout`` or ``Convolution``.
            * For output layers, you can use the following layer types:
              ``Linear``, ``Softmax`` or ``Gaussian``.

        It's possible to mix and match any of the layer types, though most often
        you should probably use hidden and output types as recommended here.  Typically,
        the last entry in this ``layers`` list should contain ``Linear`` for regression,
        or ``Softmax`` for classification.

    random_state: int
        Seed for the initialization of the neural network parameters (e.g.
        weights and biases).  This is fully deterministic.

    learning_rule: str
        Name of the learning rule used during stochastic gradient descent,
        one of ``sgd``, ``momentum``, ``nesterov``, ``adadelta`` or ``rmsprop``
        at the moment.

    learning_rate: float
        Real number indicating the default/starting rate of adjustment for
        the weights during gradient descent.  Different learning rules may
        take this into account differently.

    learning_momentum: float
        Real number indicating the momentum factor to be used for the
        learning rule 'momentum'.

    batch_size: int
        Number of training samples to group together when performing stochastic
        gradient descent.  By default each sample is treated on its own.

    n_iter: int
        The number of iterations of gradient descent to perform on the
        neural network's weights when training with ``fit()``.

    valid_set: tuple of array-like
        Validation set (X_v, y_v) to be used explicitly while training.  Both
        arrays should have the same size for the first dimention, and the second
        dimention should match with the training data specified in ``fit()``.

    valid_size: float
        Ratio of the training data to be used for validation.  0.0 means no
        validation, and 1.0 would mean there's no training data!  Common values are
        0.1 or 0.25.

    n_stable: int
        Number of interations after which training should return when the validation
        error remains constant.  This is a sign that the data has been fitted.

    f_stable: float
        Threshold under which the validation error change is assumed to be stable, to
        be used in combination with `n_stable`.

    dropout: bool or float
        Whether to use drop-out training for the inputs (jittering) and the
        hidden layers, for each training example. If a float is specified, that
        ratio of inputs will be randomly excluded during training (e.g. 0.5).

    verbose: bool
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
            verbose=False,
            **params):

        self.layers = []
        for i, layer in enumerate(layers):
            assert isinstance(layer, Layer),\
                "Specify each layer as an instance of a `sknn.mlp.Layer` object."

            # Layer names are optional, if not specified then generate one.
            if layer.name is None:
                label = "hidden" if i < len(layers)-1 else "output"
                layer.name = "%s%i" % (label, i)

            # sklearn may pass layers in as additional named parameters, remove them.
            if layer.name in params:
                del params[layer.name]

            self.layers.append(layer)

        # Don't support any additional parameters that are not in the constructor.
        # These are specified only so `get_params()` can return named layers, for double-
        # underscore syntax to work.
        assert len(params) == 0,\
            "The specified additional parameters are unknown."

        self.random_state = random_state
        self.learning_rule = learning_rule
        self.learning_rate = learning_rate
        self.learning_momentum = learning_momentum
        self.dropout = dropout if type(dropout) is float else (0.5 if dropout else 0.0)
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
            self._learning_rule = lr.AdaDelta()
        elif learning_rule == 'momentum':
            self._learning_rule = lr.Momentum(learning_momentum)
        elif learning_rule == 'nesterov':
            self._learning_rule = lr.Momentum(learning_momentum, nesterov_momentum=True)
        elif learning_rule == 'rmsprop':
            self._learning_rule = lr.RMSProp()
        else:
            raise NotImplementedError(
                "Learning rule type `%s` is not supported." % learning_rule)

        self._setup()

    def _setup(self):
        # raise NotImplementedError("MultiLayerPerceptron is an abstract class; "
        #                           "use the Classifier or Regressor instead.")
        pass

    def _create_trainer(self, dataset):
        sgd.log.setLevel(logging.WARNING)

        # Aggregate all the dropout parameters into shared dictionaries.
        probs, scales = {}, {}
        for l in [l for l in self.layers if l.dropout is not None]:
            incl = 1.0 - l.dropout
            probs[l.name] = incl
            scales[l.name] = 1.0 / incl

        if self.cost == "Dropout" or len(probs) > 0:
            # Use the globally specified dropout rate when there are no layer-specific ones.
            incl = 1.0 - self.dropout
            default_prob, default_scale = incl, 1.0 / incl

            # Pass all the parameters to pylearn2 as a custom cost function.
            self.cost = dropout.Dropout(
                default_input_include_prob=default_prob,
                default_input_scale=default_scale,
                input_include_probs=probs, input_scales=scales)

        logging.getLogger('pylearn2.monitor').setLevel(logging.WARNING)
        if dataset is not None:
            termination_criterion = tc.MonitorBased(
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

    def _check_layer(self, layer, required, optional=[]):
        required.extend(['name', 'type'])
        for r in required:
            if getattr(layer, r) is None:
                raise ValueError("Layer type `%s` requires parameter `%s`."\
                                 % (layer.type, r))

        optional.extend(['dropout'])
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
                irange=irange)

        if layer.type == 'Gaussian':
            self._check_layer(layer, ['units'])
            return mlp.LinearGaussian(
                layer_name=layer.name,
                init_beta=0.1,
                min_beta=0.001,
                max_beta=1000,
                beta_lr_scale=None,
                dim=layer.units,
                irange=irange)

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
            nvis=None if self.is_convolution else self.unit_counts[0],
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

    def _create_matrix_input(self, X, y):
        if self.is_convolution:
            # Using `b01c` arrangement of data, see this for details:
            #   http://benanne.github.io/2014/04/03/faster-convolutions-in-theano.html
            # input: (batch size, channels, rows, columns)
            # filters: (number of filters, channels, rows, columns)
            input_space = space.Conv2DSpace(shape=X.shape[1:3], num_channels=X.shape[-1])
            view = input_space.get_origin_batch(X.shape[0])
            return datasets.DenseDesignMatrix(topo_view=view, y=y), input_space
        else:
            if all([isinstance(a, numpy.ndarray) for a in (X, y)]):
                return datasets.DenseDesignMatrix(X=X, y=y), None
            else:
                return SparseDesignMatrix(X=X, y=y), None

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
        return isinstance(self.layers[0], Convolution)

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

        if self.verbose:
            log.debug("\nEpoch    Validation Error    Time"
                      "\n---------------------------------")

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
            assert self.layers[-1].units is not None,\
                "You must specify the number of units to predict without fitting."
            log.warning("Computing estimates with an untrained network.")

            self._create_specs(X)
            self._create_mlp()

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
