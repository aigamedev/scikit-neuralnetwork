Customizing Learning
====================

Training Callbacks
------------------

You have full access to — and some control over — the internal mechanism of the training algorithm via callback functions.  There are six callbacks available:
        
    * ``on_train_start`` — Called when the main training function is entered.
    * ``on_epoch_start`` — Called the first thing when a new iteration starts.
    * ``on_batch_start`` — Called before an individual batch is processed.
    * ``on_batch_finish`` — Called after that individual batch is processed.
    * ``on_epoch_finish`` — Called the first last when the iteration is done.
    * ``on_train_finish`` — Called just before the training function exits.
        
You can register for callbacks with a single function, for example:

.. code:: python

    def my_callback(event, **variables):
        print(event)        # The name of the event, as shown in the list above.
        print(variables)    # Full dictionary of local variables from training loop.

    nn = Regressor(layers=[Layer("Linear")],
                   callback=my_callback)

This function will get called for each event, which may be thousands of times depending on your dataset size. An easier way to proceed would be to use specialized callbacks.  For example, you can use callbacks on each epoch to mutate or jitter the data for training, or inject new data lazily as it is loaded.

.. code:: python

    def prepare_data(X, y, **other):
        # X and y are variables in the training code. Modify them
        # here to use new data for the next epoch.
        X[:] = X_new
        y[:] = y_new

    nn = Regressor(layers=[Layer("Linear")],
                   callback={'on_epoch_start': prepare_data})

This callback will only get triggered at the start of each epoch, before any of the data in the set has been processed.  You can also prepare the data separately in a thread and inject it into the training loop at the last minute.


Epoch Statistics
----------------

You can access statistics from the training by using another callback, specifically ``on_epoch_finish``. There, multiple variables are accessible including ``avg_valid_error`` and ``avg_train_error`` which contain the mean squared error of the last epoch, but you can also access the best results so far via ``best_valid_error`` and ``best_train_error``. 

.. code:: python

    errors = []
    def store_stats(avg_valid_error, avg_train_error, **_):
        errors.append((avg_valid_error, avg_train_error))

    nn = Classifier(
        layers=[Layer("Softmax")], n_iter=5,
        callback={'on_epoch_finish': store_stats})

After the training, you can then plot the content of the ``errors`` variable using your favorite graphing library.
