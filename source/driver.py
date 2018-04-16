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
from sklearn.utils import class_weight
from sklearn.feature_extraction.text import TfidfVectorizer

import time


def main():
    # train on subsampled data, test on subset of data
    X, y = util.get_data2('../data/subset.csv')
    vect = TfidfVectorizer(max_features=10000, min_df=2)
    X_dtm = vect.fit_transform(X)
    print(X_dtm.shape, y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X_dtm, y, test_size=0.2)

    # X2, y2 = util.get_data('../data/subsample.csv')

    baseline = DummyClassifier(strategy='stratified')
    baseline.fit(X_train, y_train)
    y_pred = baseline.predict(X_test)
    print("baseline accuracy ", metrics.accuracy_score(y_true=y_test, y_pred=y_pred))
    print("baseline precision ", metrics.precision_score(y_true=y_test, y_pred=y_pred))
    print("baseline recall ", metrics.recall_score(y_true=y_test, y_pred=y_pred))

    # C_range = 10.0 ** np.arange(-3, 3)
    C_range = [0.1]
    for i in range(len(C_range)):
        beta = 2
        t1 = time.time()
        # weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
        # cw = {0: weights[0], 1: weights[1]}
        linear_svm = SVC(kernel='linear', C=C_range[i], class_weight='balanced')
        linear_svm.fit(X_train, y_train)

        print('-------------------------')
        print("C: ", C_range[i])
        print('training took ' + str(time.time() - t1) + ' seconds')
        y_pred = linear_svm.predict(X_test)
        precision = metrics.precision_score(y_true=y_test, y_pred=y_pred)
        recall = metrics.recall_score(y_true=y_test, y_pred=y_pred)
        f1 = precision * recall / (precision + recall)
        fbeta = (1+beta**2)*precision*recall/(beta**2 * precision + recall)
        print("linear svm accuracy ", metrics.accuracy_score(y_true=y_test, y_pred=y_pred))
        print("linear svm precision ", precision)
        print("linear recall ", recall)
        print("linear svm f1 score", f1)
        print("linear svm fbeta score", fbeta)

    # C_range = np.logspace(-2, 3, 6)
    # gamma_range = np.logspace(-4, 1, 6)
    # C_range = [0.1]
    # gamma_range = [10]
    # for i in range(len(C_range)):
    #     for j in range(len(gamma_range)):
    #         beta = 2
    #         t1 = time.time()
    #         # weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
    #         # cw = {0: weights[0], 1: weights[1]}
    #         linear_svm = SVC(kernel='linear', C=C_range[i], gamma=gamma_range[j], class_weight='balanced')
    #         linear_svm.fit(X_train, y_train)
    #
    #         print('-------------------------')
    #         print("C: ", C_range[i], "gamma: ", gamma_range[j])
    #         print('training took ' + str(time.time() - t1) + ' seconds')
    #         y_pred = linear_svm.predict(X_test)
    #         precision = metrics.precision_score(y_true=y_test, y_pred=y_pred)
    #         recall = metrics.recall_score(y_true=y_test, y_pred=y_pred)
    #         f1 = precision * recall / (precision + recall)
    #         fbeta = (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall)
    #         print("rbf svm accuracy ", metrics.accuracy_score(y_true=y_test, y_pred=y_pred))
    #         print("rbf svm precision ", precision)
    #         print("rbf recall ", recall)
    #         print("rbf svm f1 score", f1)
    #         print("rbf svm fbeta score", fbeta)


if __name__ == '__main__':
    main()
