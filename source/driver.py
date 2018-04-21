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
import matplotlib.pyplot as plt

import time


def main():
    X, y = util.get_data2('../data/subset.csv')

    # split on tfidf, then only select columns
    vect = TfidfVectorizer(max_features=None, min_df=2)
    X_dtm = vect.fit_transform(X)
    print("old shape", X_dtm.shape)

    # finding info gains
    t1 = time.time()
    info_gains = np.apply_along_axis(util.info_gain, 0, X_dtm.toarray(), y, 0.00001)
    print("took", time.time() - t1, 'seconds')

    # # printing vocab
    # vocab = vect.vocabulary_
    # inv_map = {v: k for k, v in vocab.iteritems()}
    # for i in range(len(max_cols)):
    #     print(inv_map[max_cols[i]])

    # # plotting for tfidf threshold
    # zeros = np.where(y == 0)[0]
    # ones = np.where(y == 1)[0]
    # response = vect.transform(X[zeros])
    # a = response.toarray().flatten()[::-1]
    # plt.hist(a,  np.arange(0.00001, 1.01, 0.01))
    # plt.show()

    num_features = np.arange(10000, 2000, -500)
    for f in num_features:
        max_cols = info_gains.argsort()[-f:][::-1]
        X = X_dtm[:, max_cols]
        print("new shape", X.shape)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
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
            f1 = 2 * precision * recall / (precision + recall)
            fbeta = (1+beta**2)*precision*recall/(beta**2 * precision + recall)
            print("linear svm accuracy ", metrics.accuracy_score(y_true=y_test, y_pred=y_pred))
            print("linear svm precision ", precision)
            print("linear recall ", recall)
            print("linear svm f1 score", f1)
            print("linear svm fbeta score", fbeta)

        C_range = np.logspace(-2, 3, 6)
        gamma_range = np.logspace(-4, 1, 6)
        C_range = [0.1]
        gamma_range = [10]
        for i in range(len(C_range)):
            for j in range(len(gamma_range)):
                beta = 2
                t1 = time.time()
                # weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
                # cw = {0: weights[0], 1: weights[1]}
                rbf_svm = SVC(kernel='rbf', C=C_range[i], gamma=gamma_range[j], class_weight='balanced')
                rbf_svm.fit(X_train, y_train)

                print('-------------------------')
                print("C: ", C_range[i], "gamma: ", gamma_range[j])
                print('training took ' + str(time.time() - t1) + ' seconds')
                y_pred = rbf_svm.predict(X_test)
                precision = metrics.precision_score(y_true=y_test, y_pred=y_pred)
                recall = metrics.recall_score(y_true=y_test, y_pred=y_pred)
                f1 = 2 * precision * recall / (precision + recall)
                fbeta = (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall)
                print("rbf svm accuracy ", metrics.accuracy_score(y_true=y_test, y_pred=y_pred))
                print("rbf svm precision ", precision)
                print("rbf recall ", recall)
                print("rbf svm f1 score", f1)
                print("rbf svm fbeta score", fbeta)
                print('===========================', '\n')


if __name__ == '__main__':
    main()
