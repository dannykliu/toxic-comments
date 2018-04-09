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
    X, y = util.get_data('../data/subsample.csv')
    print X.shape, y.shape

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    baseline = DummyClassifier(strategy='most_frequent')
    baseline.fit(X_train, y_train)

    y_pred = baseline.predict(X_test)
    print "baseline accuracy test: ", metrics.accuracy_score(y_test, y_pred)
    print "baseline accuracy train: ", metrics.accuracy_score(baseline.predict(X_train), y_train)

    t1 = time.time()
    linear_svm = SVC(kernel='linear')
    linear_svm.fit(X_train, y_train)
    print 'training took ' + str(time.time() - t1) + ' seconds'

    # y_pred = linear_svm.predict(X_test)
    # print "linear svm accuracy: ", metrics.accuracy_score(y_test, y_pred)
    # print "linear svm recall: ", metrics.recall_score(y_test, y_pred)
    # print "linear svm precision: ", metrics.precision_score(y_test, y_pred)

    y_pred = linear_svm.predict(X_train)
    print "linear svm accuracy: ", metrics.accuracy_score(y_train, y_pred)
    print "linear svm recall: ", metrics.recall_score(y_train, y_pred)
    print "linear svm precision: ", metrics.precision_score(y_train, y_pred)


if __name__ == '__main__':
    main()
