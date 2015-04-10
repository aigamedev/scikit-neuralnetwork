from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_mldata
import logging
import sys
import numpy as np

np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)

log = logging.getLogger()
log.setLevel(logging.DEBUG)

stdout = logging.StreamHandler(sys.stdout)
stdout.setLevel(logging.DEBUG)
stdout.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
log.addHandler(stdout)

mnist = fetch_mldata('MNIST original')
X_train, X_test, y_train, y_test = train_test_split(mnist.data / 255.0, mnist.target, test_size=0.33, random_state=1234)

classifiers = []




# try:
#     raise ImportError
#
#     from nolearn.dbn import DBN
#     clf = DBN(
#         [X_train.shape[1], 300, 10],
#         learn_rates=0.3,
#         learn_rate_decays=0.9,
#         epochs=10,
#         verbose=1)
#     classifiers.append(('nolearn.dbn', clf))
# except ImportError:
#     pass


try:
    from sknn.mlp import MultiLayerPerceptronClassifier

    clf = MultiLayerPerceptronClassifier(
        layers=[("Maxout", 50, 2),("Maxout", 50, 2), ("Softmax",)],
        learning_rate=0.01,
        batch_size=10,
        n_stable=50,
        f_stable=0.0,
        n_iter=2,
        learning_rule='momentum',
        # dropout=True,
        verbose=1,
    )
    classifiers.append(('sknn.mlp', clf))
except ImportError:
    pass


for name, clf in classifiers:
    print y_train

    clf.fit(X_train, y_train)

    from sklearn.metrics import classification_report

    y_pred = clf.predict(X_test)
    print name
    #exit()
    print "\tAccuracy:", clf.score(X_test, y_test)
    print "\tReport:"
    print classification_report(y_test, y_pred)
