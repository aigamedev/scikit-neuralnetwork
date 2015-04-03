# Based on scikit-learn examples.
#   - File: auto_examples/ensemble/plot_adaboost_regression.py
#   - Author: Noel Dawe <noel.dawe@gmail.com>

import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

from sknn.mlp import MultiLayerPerceptronRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import logging
import sys

log = logging.getLogger()
log.setLevel(logging.DEBUG)

stdout = logging.StreamHandler(sys.stdout)
stdout.setLevel(logging.DEBUG)
stdout.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
log.addHandler(stdout)

# Preparation.
rng = np.random.RandomState(1)
X = np.linspace(0, 1, 100)[:, np.newaxis]
y = (np.sin(X*6).ravel() + np.sin(X*36).ravel() + rng.normal(0, 0.1, X.shape[0])) / 3.0

# Construction.
clf_1 = DecisionTreeRegressor(max_depth=4)
clf_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),
                          n_estimators=300, random_state=rng)
clf_3 = MultiLayerPerceptronRegressor(layers=[("Linear",)], n_iter=100)
# clf_4 = MultiLayerPerceptronRegressor(
#             layers=[("Maxout", 20, 2),("Maxout", 20, 2),("Maxout", 20, 2),  ("Linear",)],
#             learning_rate=0.001, learning_rule="rmsprop", dropout=False, batch_size=10, n_iter=1000)


clf_4 = MultiLayerPerceptronRegressor(
            layers=[("Maxout", 20, 2),("Maxout", 20, 2),("Maxout", 20, 2),  ("Linear",)],
            learning_rate=0.01, learning_rule="momentum", dropout=False, batch_size=10, n_iter=500, verbose=True)


#clf_4 = make_pipeline(StandardScaler(), clf_4)
clf_4 = make_pipeline(MinMaxScaler(feature_range=(-1,1)), clf_4)


# Training.
clf_1.fit(X, y)
clf_2.fit(X, y)
clf_3.fit(X, y)
clf_4.fit(X, y)

# Prediction.
y_1 = clf_1.predict(X)
y_2 = clf_2.predict(X)
y_3 = clf_3.predict(X)
y_4 = clf_4.predict(X)

# Plotting.
plt.figure()
plt.scatter(X, y, c="k", label="training samples")
plt.plot(X, y_1, c="r", label="decision tree", linewidth=2)
plt.plot(X, y_2, c="r", label="tree ensemble", linewidth=1)
plt.plot(X, y_3, c="b", label="linear network", linewidth=1)
plt.plot(X, y_4, c="b", label="deep network", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Machine Learning Comparison")
plt.legend()
plt.show()
