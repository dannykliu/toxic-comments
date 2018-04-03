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
    print X.shape, y.shape

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    baseline = DummyClassifier(strategy='most_frequent')
    baseline.fit(X, y)
    print metrics.accuracy_score(baseline.predict(X_test), y_test)

    t1 = time.time()
    linear_svm = SVC(kernel='linear', C=1.0)
    linear_svm.fit(X, y)

    print 'training took ' + str(time.time() - t1) + ' seconds'
    print metrics.accuracy_score(linear_svm.predict(X_test), y_test)


if __name__ == '__main__':
    main()
