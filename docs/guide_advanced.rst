Advanced Usage
==============

The examples in this section help you get more out of ``scikit-neuralnetwork`` via integration or configuration of the other libraries it works with, such as Theano or scikit-learn.


GPU Backend
-----------

To setup the library to use your GPU or CPU explicitly in 32-bit or 64-bit mode, you can use the ``backend`` pseudo-module.  It's a syntactic helper to setup ``THEANO_FLAGS`` in a Pythonic way, for example:

.. code:: python

    # Use the GPU in 32-bit mode, falling back otherwise.
    from sknn.backend import gpu32
    
    # Use the CPU in 64-bit mode.
    from sknn.backend import cpu64
    

WARNING: This will only work if your program has not yet imported the ``theano`` module, due to the way the library is designed.  If ``THEANO_FLAGS`` are set on the command-line, they are not overwridden.


.. _example-pipeline:

Pipeline
--------

Typically, neural networks perform better when their inputs have been normalized or standardized.  Using a scikit-learn's `pipeline <http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html>`_ support is an obvious choice to do this.

Here's how to setup such a pipeline with a multi-layer perceptron as a classifier:

.. code:: python

    from sknn.mlp improt Classifier, Layer

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import MinMaxScaler

    pipeline = Pipeline([
            ('min/max scaler', MinMaxScaler(feature_range=(0.0, 1.0))),
            ('neural network', Classifier(layers=[Layer("Softmax")], n_iter=25))])
    pipeline.fit(X_train, y_train)

You can then use the pipeline as you would the neural network, or any other standard API from scikit-learn.


Grid Search
-----------

In scikit-learn, you can use a ``GridSearchCV`` to optimize your neural network's hyper-parameters automatically, both the top-level parameters and the parameters within the layers.  For example, assuming you have your MLP constructed as in the :ref:`example-regression` example in the local variable called ``nn``, the layers are named automatically so you can refer to them as follows:

    * ``hidden0``
    * ``hidden1``
    * ...
    * ``output``
     
Keep in mind you can manually specify the ``name`` of any ``Layer`` in the constructor if you don't want the automatically assigned name.  Then, you can use sklearn's hierarchical parameters to perform a grid search over those nested parameters too: 

.. code:: python

    from sklearn.grid_search import GridSearchCV

    gs = GridSearchCV(nn, param_grid={
        learning_rate: [0.05, 0.01, 0.005, 0.001],
        'hidden0__units': [4, 8, 12],
        'hidden0__type': ["Rectifier", "Sigmoid", "Tanh"]})
    gs.fit(a_in, a_out)
    
This will search through the listed ``learning_rate`` values, the number of hidden units and the activation type for that layer too, and find the best combination of parameters.


Randomized Search
-----------------

In the cases when you have large numbers of hyper-parameters that you want to try automatically to find a good combination, you can use a randomized search as follows:

.. code:: python

    from scipy import stats
    from sklearn.grid_search import RandomizedSearchCV

    rs = RandomizedSearchCV(nn, param_grid={
        learning_rate: stats.uniform(0.001, 0.05),
        'hidden0__units': stats.randint(4, 12),
        'hidden0__type': ["Rectifier", "Sigmoid", "Tanh"]})
    rs.fit(a_in, a_out)

This works for both :class:`sknn.mlp.Classifier` and :class:`sknn.mlp.Regressor`.