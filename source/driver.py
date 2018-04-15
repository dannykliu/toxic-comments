from __future__ import print_function

import util
import numpy as np

# scikit-learn libraries
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import time


def main():
    # train on subsampled data, test on subset of data
    X, y = util.get_data('../data/subset.csv')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    print(X.shape, y.shape)

    baseline = DummyClassifier(strategy='uniform')
    baseline.fit(X_train, y_train)
    y_pred = baseline.predict(X_test)
    print("baseline accuracy ", metrics.accuracy_score(y_true=y_test, y_pred=y_pred))
    print("baseline precision ", metrics.precision_score(y_true=y_test, y_pred=y_pred))
    print("baseline recall ", metrics.recall_score(y_true=y_test, y_pred=y_pred))

    t1 = time.time()
    linear_svm = SVC(kernel='linear', C=1.0)
    linear_svm.fit(X_train, y_train)
    print('training took ' + str(time.time() - t1) + ' seconds')

    y_pred = linear_svm.predict(X_test)
    print("linear svm accuracy ", metrics.accuracy_score(y_true=y_test, y_pred=y_pred))
    print("linear svm precision ", metrics.precision_score(y_true=y_test, y_pred=y_pred))
    print("linear recall ", metrics.recall_score(y_true=y_test, y_pred=y_pred))


if __name__ == '__main__':
    main()
