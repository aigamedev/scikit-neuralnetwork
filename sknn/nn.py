# -*- coding: utf-8 -*-
from __future__ import (absolute_import, unicode_literals, print_function)

__all__ = ['Regressor', 'Classifier', 'Layer', 'Convolution']

import os
import sys
import time
import logging
import itertools

log = logging.getLogger('sknn')


import numpy
import theano


class ansi:
    BOLD = '\033[1;97m'
    WHITE = '\033[0;97m'
    YELLOW = '\033[0;33m'
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    BLUE = '\033[0;94m'
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
        will then be accessible to scikit-learn via a nested sub-object.  For example,
        if name is set to ``layer1``, then the parameter ``layer1__units`` from the network
        is bound to this layer's ``units`` variable.

        The name defaults to ``hiddenN`` where N is the integer index of that layer, and the
        final layer is always ``output`` without an index.

    units: int
        The number of units (also known as neurons) in this layer.  This applies to all
        layer types except for convolution.

    pieces: int, optional
        The number of piecewise linear segments in the Maxout activation.  This is
        optional and only applies when `Maxout` is selected as the layer type.

    weight_decay: float, optional
        The coefficient for L1 or L2 regularization of the weights.  For example, a value of
        0.0001 is multiplied by the L1 or L2 weight decay equation.

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
            weight_decay=None,
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
        self.weight_decay = weight_decay
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
        return "<sknn.nn.%s `%s`: %s>" % (self.__class__.__name__, self.type, params)


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
        will then be accessible to scikit-learn via a nested sub-object.  For example,
        if name is set to ``layer1``, then the parameter ``layer1__units`` from the network
        is bound to this layer's ``units`` variable.

        The name defaults to ``hiddenN`` where N is the integer index of that layer, and the
        final layer is always ``output`` without an index.

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

    weight_decay: float, optional
        The coefficient for L1 or L2 regularization of the weights.  For example, a value of
        0.0001 is multiplied by the L1 or L2 weight decay equation.

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
            weight_decay=None,
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
                weight_decay=weight_decay,
                dropout=dropout)

        self.channels = channels
        self.pool_shape = pool_shape or (1,1)
        self.pool_type = pool_type or ('max' if pool_shape else None)
        self.kernel_shape = kernel_shape
        self.kernel_stride = kernel_stride or self.pool_shape
        self.border_mode = border_mode


class NeuralNetwork(object):
    """
    Abstract base class for wrapping all neural network functionality from PyLearn2,
    common to multi-layer perceptrons in :module:`sknn.mlp` and auto-encoders in
    in :module:`sknn.ae`.

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
        one of ``sgd``, ``momentum``, ``nesterov``, ``adadelta``, ``adagrad`` or
        ``rmsprop`` at the moment.  The default is vanilla ``sgd``.

    learning_rate: float
        Real number indicating the default/starting rate of adjustment for
        the weights during gradient descent.  Different learning rules may
        take this into account differently.

    learning_momentum: float
        Real number indicating the momentum factor to be used for the
        learning rule 'momentum'.

    batch_size: int
        Number of training samples to group together when performing stochastic
        gradient descent (technically, a "minibatch").  By default each sample is
        treated on its own, with ``batch_size=1``.  Larger batches are usually faster.

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

    regularize: string, optional
        Which regularization technique to use on the weights, for example ``L2`` (most
        common) or ``L1`` (quite rare), as well as ``dropout``.  By default, there's no
        regularization, unless another parameter implies it should be enabled, e.g. if
        ``weight_decay`` or ``dropout_rate`` are specified.

    weight_decay: float, optional
        The coefficient used to multiply either ``L1`` or ``L2`` equations when computing
        the weight decay for regularization.  If ``regularize`` is specified, this defaults
        to 0.0001.
        
    dropout_rate: float, optional
        What rate to use for drop-out training in the inputs (jittering) and the
        hidden layers, for each training example. Specify this as a ratio of inputs
        to be randomly excluded during training, e.g. 0.75 means only 25% of inputs
        will be included in the training.

    loss_type: string, optional
        The cost function to use when training the network.  There are two valid options:

            * ``mse`` — Use mean squared error, for learning to predict the mean of the data.
            * ``mae`` — Use mean average error, for learning to predict the median of the data.

        The default option is ``mse``, and ``mae`` can only be applied to layers of type
        ``Linear`` or ``Gaussian`` and they must be used as the output layer.

    debug: bool, optional
        Should the underlying training algorithms perform validation on the data
        as it's optimizing the model?  This makes things slower, but errors can
        be caught more effectively.  Default is off.

    verbose: bool, optional
        How to initialize the logging to display the results during training. If there is
        already a logger initialized, either ``sknn`` or the root logger, then this function
        does nothing.  Otherwise:

            * ``False`` — Setup new logger that shows only warnings and errors.
            * ``True`` — Setup a new logger that displays all debug messages.
            * ``None`` — Don't setup a new logger under any condition (default). 

        Using the built-in python ``logging`` module, you can control the detail and style of
        output by customising the verbosity level and formatter for ``sknn`` logger.
    """

    def __init__(
            self,
            layers,
            random_state=None,
            learning_rule='sgd',
            learning_rate=0.01,
            learning_momentum=0.9,
            regularize=None,
            weight_decay=None,
            dropout_rate=None,
            batch_size=1,
            n_iter=None,
            n_stable=50,
            f_stable=0.001,
            valid_set=None,
            valid_size=0.0,
            loss_type='mse',
            debug=False,
            verbose=None,
            **params):

        self.layers = []
        for i, layer in enumerate(layers):
            assert isinstance(layer, Layer),\
                "Specify each layer as an instance of a `sknn.mlp.Layer` object."

            # Layer names are optional, if not specified then generate one.
            if layer.name is None:
                layer.name = ("hidden%i" % i) if i < len(layers)-1 else "output"

            # sklearn may pass layers in as additional named parameters, remove them.
            if layer.name in params:
                del params[layer.name]

            self.layers.append(layer)

        # Don't support any additional parameters that are not in the constructor.
        # These are specified only so `get_params()` can return named layers, for double-
        # underscore syntax to work.
        assert len(params) == 0,\
            "The specified additional parameters are unknown: %s." % ','.join(params.keys())

        # Basic checking of the freeform string options.
        assert regularize in (None, 'L1', 'L2', 'dropout'),\
            "Unknown type of regularization specified: %s." % self.regularize
        assert loss_type in ('mse', 'mae'),\
            "Unknown loss function type specified: %s." % self.loss_type

        self.random_state = random_state
        self.learning_rule = learning_rule
        self.learning_rate = learning_rate
        self.learning_momentum = learning_momentum
        self.regularize = regularize or ('dropout' if dropout_rate else None)\
                                     or ('L2' if weight_decay else None)
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.n_stable = n_stable
        self.f_stable = f_stable
        self.valid_set = valid_set
        self.valid_size = valid_size
        self.loss_type = loss_type
        self.debug = debug
        self.verbose = verbose
        self.weights = None
        
        self._backend = None
        self._create_logger()

    @property
    def is_initialized(self):
        """Check if the neural network was setup already.
        """
        return self._backend is not None and self._backend.is_initialized
        raise NotImplementedError("NeuralNetwork is an abstract class; "
                                  "use the mlp.Classifier or mlp.Regressor instead.")

    @property
    def is_convolution(self):
        """Check whether this neural network includes convolution layers.
        """
        return isinstance(self.layers[0], Convolution)

    def _create_logger(self):
        # If users have configured logging already, assume they know best.
        if len(log.handlers) > 0 or len(log.parent.handlers) > 0 or self.verbose is None:
            return

        # Otherwise setup a default handler and formatter based on verbosity.
        lvl = logging.DEBUG if self.verbose else logging.WARNING
        fmt = logging.Formatter("%(message)s")
        hnd = logging.StreamHandler(stream=sys.stdout)

        hnd.setFormatter(fmt)
        hnd.setLevel(lvl)
        log.addHandler(hnd)
        log.setLevel(lvl)

    def _train_layer(self, trainer, layer, dataset):
        # Bug in PyLearn2 that has some unicode channels, can't sort.
        layer.monitor.channels = {str(k): v for k, v in layer.monitor.channels.items()}
        best_valid_error = float("inf")

        for i in itertools.count(1):
            start = time.time()
            trainer.train(dataset=dataset)

            layer.monitor.report_epoch()
            layer.monitor()
            
            objective = layer.monitor.channels.get('objective', None)
            if objective:
                avg_valid_error = objective.val_shared.get_value()
                best_valid_error = min(best_valid_error, avg_valid_error)
            else:
                # 'objective' channel is only defined with validation set.
                avg_valid_error = None

            best_valid = bool(best_valid_error == avg_valid_error)
            log.debug("{:>5}      {}{}{}        {:>3.1f}s".format(
                      i,
                      ansi.GREEN if best_valid else "",
                      "{:>10.6f}".format(float(avg_valid_error)) if avg_valid_error else "     N/A  ",
                      ansi.ENDC if best_valid else "",
                      time.time() - start
                      ))

            if not trainer.continue_learning(layer):
                log.debug("")
                log.info("Early termination condition fired at %i iterations.", i)
                break
            if self.n_iter is not None and i >= self.n_iter:
                log.debug("")
                log.info("Terminating after specified %i total iterations.", i)
                break
