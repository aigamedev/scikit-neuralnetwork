from __future__ import (absolute_import, unicode_literals)

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
from pylearn2.training_algorithms.learning_rule import RMSProp, Momentum
from pylearn2.space import Conv2DSpace
from pylearn2.termination_criteria import MonitorBased


class ansi:
    BOLD = '\033[1;97m'
    WHITE = '\033[0;97m'
    BLUE = '\033[0;94m'
    GREEN = '\033[0;32m'
    ENDC = '\033[0m'


class BaseMLP(sklearn.base.BaseEstimator):
    """
    A wrapper for PyLearn2 compatible with scikit-learn.

    Parameters
    ----------
    layers : list of tuples
        An iterable sequence of each layer each as a tuple: first with an
        activation type and then optional parameters such as the number of
        units.

    random_state : int
        Seed for the initialization of the neural network parameters (e.g.
        weights and biases).  This is fully deterministic.

    learning_rule : str
        Name of the learning rule used during stochastic gradient descent,
        one of ('sgd', 'momentum', 'rmsprop') at the moment.    

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
        neural network's weights when training with fit().

    dropout : bool
        Whether to use drop-out training for the inputs (jittering) and the
        hidden layers, for each training example.

    verbose : bool
        If True, print the score at each epoch.
    """

    def __init__(
            self,
            layers,
            random_state=None,
            learning_rule='sgd',
            learning_rate=0.01,
            learning_momentum=0.9,
            batch_size=1,
            n_iter=None,
            valid_set=None,
            valid_size=0.0,
            n_stable=50,
            f_stable=0.001,
            dropout=False,
            verbose=False):

        self.layers = layers
        self.seed = random_state
        self.verbose = verbose

        self.unit_counts = None
        self.mlp = None
        self.ds = None
        self.trainer = None
        self.f = None
        self.train_set = None

        self.cost = "Dropout" if dropout else None
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.n_stable = n_stable
        self.f_stable = f_stable
        self.valid_set = valid_set
        self.valid_size = valid_size

        self.best_valid_error = float("inf")

        if learning_rule == 'sgd':
            self.learning_rule = None
        elif learning_rule == 'momentum':
            self.learning_rule = Momentum(learning_momentum)
        elif learning_rule == 'rmsprop':
            self.learning_rule = RMSProp()
        else:
            raise NotImplementedError(
                "Learning rule type `%s` is not supported." % learning_rule)

        self._setup()

    def _setup(self):
        pass

    def _create_trainer(self, dataset):
        sgd.log.setLevel(logging.WARNING)

        if self.cost == "Dropout":
            first_hidden_name = "Hidden_0_"+self.layers[0][0]
            self.cost = Dropout(
                input_include_probs={first_hidden_name: 1.0},
                input_scales={first_hidden_name: 1.})

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
            learning_rule=self.learning_rule,
            learning_rate=self.learning_rate,
            termination_criterion=termination_criterion,
            monitoring_dataset=dataset)

    def _create_hidden_layer(self, name, args, irange=0.1):
        activation_type = args[0]
        if activation_type == "Rectifier":
            return mlp.RectifiedLinear(
                layer_name=name,
                dim=args[1],
                irange=irange)

        if activation_type == "Sigmoid":
            return mlp.Sigmoid(
                layer_name=name,
                dim=args[1],
                irange=irange)

        if activation_type == "Tanh":
            return mlp.Tanh(
                layer_name=name,
                dim=args[1],
                irange=irange)

        if activation_type == "Maxout":
            return maxout.Maxout(
                layer_name=name,
                num_units=args[1],
                num_pieces=args[2],
                irange=irange)

        if activation_type == "Convolution":
            return mlp.ConvRectifiedLinear(
                layer_name=name,
                output_channels=args[1],
                kernel_shape=args[2],
                pool_shape=(1,1),
                pool_stride=(1,1),
                irange=irange)

        raise NotImplementedError(
            "Hidden layer type `%s` is not implemented." % activation_type)

    def _create_output_layer(self, name, args):
        activation_type = args[0]
        if activation_type == "Linear":
            return mlp.Linear(
                dim=args[1],
                layer_name=name,
                irange=0.00001)

        if activation_type == "LinearGaussian":
            return mlp.LinearGaussian(
                init_beta=0.1,
                min_beta=0.001,
                max_beta=1000,
                beta_lr_scale=None,
                dim=args[1],
                layer_name=name,
                irange=0.1)

        if activation_type == "Softmax":
            return mlp.Softmax(
                layer_name=name,
                n_classes=args[1],
                irange=0.1)

        raise NotImplementedError(
            "Output layer type `%s` is not implemented." % activation_type)

    def _create_mlp(self, X, y, nvis=None, input_space=None):
        # Create the layers one by one, connecting to previous.
        mlp_layers = []
        for i, layer in enumerate(self.layers[:-1]):
            fan_in = self.unit_counts[i] + 1
            fan_out = self.unit_counts[i + 1]
            lim = numpy.sqrt(6) / (numpy.sqrt(fan_in + fan_out))

            layer_name = "Hidden_%i_%s" % (i, layer[0])
            hidden_layer = self._create_hidden_layer(layer_name, layer, irange=lim)
            mlp_layers.append(hidden_layer)

        # Deal with output layer as a special case.
        output_layer_info = list(self.layers[-1])
        output_layer_info.append(self.unit_counts[-1])

        output_layer_name = "Output_%s" % output_layer_info[0]
        output_layer = self._create_output_layer(output_layer_name, output_layer_info)
        mlp_layers.append(output_layer)

        return mlp.MLP(
            mlp_layers,
            nvis=nvis,
            seed=self.seed,
            input_space=input_space)

    def _initialize(self, X, y):
        assert not self.is_initialized,\
            "This neural network has already been initialized."

        log.info(
            "Initializing neural network with %i layers, %i inputs and %i outputs.",
            len(self.layers), X.shape[1], y.shape[1])

        # Calculate and store all layer sizes.
        self.layers[-1] = (self.layers[-1][0], y.shape[1])
        self.unit_counts = [X.shape[1]]
        for layer in self.layers:
            self.unit_counts += [layer[1]]
            log.debug("  - Type: {}{: <10}{}  Units: {}{: <4}{}".format(
                ansi.BOLD, layer[0], ansi.ENDC, ansi.BOLD, layer[1], ansi.ENDC))
        log.debug("")

        if self.valid_size > 0.0:
            assert self.valid_set is None, "Can't specify valid_size and valid_set together."
            X, X_v, y, y_v = sklearn.cross_validation.train_test_split(
                                X, y,
                                test_size=self.valid_size,
                                random_state=self.seed)
            self.valid_set = X_v, y_v
        self.train_set = X, y

        # Convolution networks need a custom input space.
        if self.is_convolution:
            nvis = None
            input_space = Conv2DSpace(shape=X.shape[1:], num_channels=1)
            view = input_space.get_origin_batch(100)
            self.ds = DenseDesignMatrix(topo_view=view, y=y)
        else:
            nvis = self.unit_counts[0]
            input_space = None
            self.ds = DenseDesignMatrix(X=X, y=y)

        if self.valid_set:
            X_v, y_v = self.valid_set
            self.vs = DenseDesignMatrix(X=X_v, y=y_v)
        else:
            self.vs = None

        if self.mlp is None:
            self.mlp = self._create_mlp(X, y, input_space=input_space, nvis=nvis)

        self.trainer = self._create_trainer(self.vs)
        self.trainer.setup(self.mlp, self.ds)
        inputs = self.mlp.get_input_space().make_theano_batch()
        self.f = theano.function([inputs], self.mlp.fprop(inputs))

    @property
    def is_initialized(self):
        """Check if the neural network was setup already.
        """
        return not (self.ds is None or self.trainer is None or self.f is None)

    @property
    def is_convolution(self):
        """Check whether this neural network includes convolution layers.
        """
        return "Conv" in self.layers[0][0]

    def __getstate__(self):
        assert self.mlp is not None,\
            "The neural network has not been initialized."

        d = self.__dict__.copy()
        for k in ['ds', 'f', 'trainer']:
            if k in d:
                del d[k]
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

        for k in ['ds', 'f', 'trainer']:
            setattr(self, k, None)

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
            X = numpy.array([X]).transpose(1,2,3,0)
            X = self.ds.view_converter.topo_view_to_design_mat(X)

        self.ds.X, self.ds.y = X, y

        # Bug in PyLearn2 that has some unicode channels, can't sort.
        self.mlp.monitor.channels = {str(k): v for k, v in self.mlp.monitor.channels.items()}

        log.info("Training on dataset of {:,} samples with {:,} total size.".format(num_samples, data_size))
        if self.valid_set:
            X_v, _ = self.valid_set
            log.debug("  - Test: {: <10,}  Valid: {: <4,}".format(X.shape[0], X_v.shape[0]))
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
        if self.is_convolution:
            X = numpy.array([X]).transpose(1,2,3,0)

        return self.f(X)



class MultiLayerPerceptronRegressor(BaseMLP, sklearn.base.RegressorMixin):

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

    def _setup(self):
        self.label_binarizer = sklearn.preprocessing.LabelBinarizer()

    def fit(self, X, y):
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
        # For binary case, LabelBinarizer only has one output so convert to 2D.
        if proba.shape[-1] == 1:
            p = proba.flatten()
            proba = numpy.zeros((proba.shape[0], 2))
            proba[:,0], proba[:,1] = 1.0-p, p
        # Normalize so every row sums to one.
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
