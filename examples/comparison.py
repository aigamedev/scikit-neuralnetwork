# Based on scikit-learn examples.
#   - File: auto_examples/ensemble/plot_adaboost_regression.py
#   - Author: Noel Dawe <noel.dawe@gmail.com>

import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

from sknn.nn import SimpleNeuralRegressor


# Create the dataset
rng = np.random.RandomState(1)
X = np.linspace(0, 1, 100)[:, np.newaxis]
y = (np.sin(X*6).ravel() + np.sin(X*36).ravel() + rng.normal(0, 0.1, X.shape[0])) / 3.0

# Fit regression model
clf_1 = DecisionTreeRegressor(max_depth=4)
clf_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),
                          n_estimators=300, random_state=rng)
clf_3 = SimpleNeuralRegressor(layers=[("Linear",)])
clf_4 = SimpleNeuralRegressor(layers=[("RectifiedLinear",250),("RectifiedLinear",200),
                                      ("RectifiedLinear",150),("RectifiedLinear",100),
                                      ("Linear",)],
                              learning_rate=0.1, dropout=True)

print X.shape, y.shape
clf_1.fit(X, y)
clf_2.fit(X, y)
clf_3.fit(X, y, n_iter=100)
clf_4.fit(X, y, n_iter=5000)

# Predict
y_1 = clf_1.predict(X)
y_2 = clf_2.predict(X)
y_3 = clf_3.predict(X)
y_4 = clf_4.predict(X)

# Plot the results
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
