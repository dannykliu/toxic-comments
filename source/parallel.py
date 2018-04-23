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
from sklearn import preprocessing
import multiprocessing
import Queue
import time



def trainRBF(j,i, metricsQueue, f, X_train, y_train, X_test, y_test, C_range, gamma_range):
    beta = 2
    t1 = time.time()
    # weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
    # cw = {0: weights[0], 1: weights[1]}
    rbf_svm = SVC(kernel='rbf', C=C_range[i], gamma=gamma_range[j], class_weight='balanced')
    rbf_svm.fit(X_train, y_train)

    print('-------------------------')
    print('Num features:', f)
    print("C: ", C_range[i], "gamma: ", gamma_range[j])
    print('training took ' + str(time.time() - t1) + ' seconds')
    y_pred = rbf_svm.predict(X_test)
    precision = metrics.precision_score(y_true=y_test, y_pred=y_pred)
    recall = metrics.recall_score(y_true=y_test, y_pred=y_pred)
    fbeta = (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall)
    print("rbf train accuracy ", metrics.accuracy_score(y_true=y_train, y_pred=rbf_svm.predict(X_train)))

    test_accuracy = metrics.accuracy_score(y_true=y_test, y_pred=y_pred)
    print("rbf test accuracy ", test_accuracy)
    print("rbf test precision ", precision)
    print("rbf test recall ", recall)
    print("rbf test fbeta score", fbeta)
    
    metrics.put((i, j, test_accuracy, fbeta, recall))

def main():
    X, y, raw = util.get_data('../data/subset.csv')
    # get homegrown features
    new_features = util.get_features(raw)

    # split on tfidf, then only select columns
    vect = TfidfVectorizer(max_features=None, min_df=2)
    X_dtm = vect.fit_transform(X)

    # finding info gains
    t1 = time.time()
    info_gains = np.apply_along_axis(util.info_gain, 0, X_dtm.toarray(), y, 0.0001)
    print("info gain took", time.time() - t1, 'seconds')
    max_cols = info_gains.argsort()[-2000:][::-1]

    # printing vocab
    # vocab = vect.vocabulary_
    # inv_map = {v: k for k, v in vocab.iteritems()}
    # for i in range(len(max_cols)):
    #     print(inv_map[max_cols[i]], end=' ')

    # # plotting for tfidf threshold
    # zeros = np.where(y == 0)[0]
    # ones = np.where(y == 1)[0]
    # response = vect.transform(X[zeros])
    # a = response.toarray().flatten()[::-1]
    # plt.hist(a,  np.arange(0.00001, 1.01, 0.01))
    # plt.show()

    # num_features = np.arange(500, 2000, 500)
    # num_features = np.arange(500, 2000, 200)
    num_features = [1900]
    best_accuracy = 0.0
    best_fbeta = 0.0
    best_recall = 0.0
    for f in num_features:
        max_cols = info_gains.argsort()[-f:][::-1]
        # turn X from sparse matrix to numpy array
        X = X_dtm[:, max_cols].toarray()
        # add our features as columns to X
        for new_feature in new_features:
            X = np.append(X, new_feature.reshape(-1, 1), axis=1)
        print("new X shape", X.shape)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
        baseline = DummyClassifier(strategy='stratified')
        baseline.fit(X_train, y_train)
        y_pred = baseline.predict(X_test)
        print("baseline accuracy ", metrics.accuracy_score(y_true=y_test, y_pred=y_pred))
        print("baseline precision ", metrics.precision_score(y_true=y_test, y_pred=y_pred))
        print("baseline recall ", metrics.recall_score(y_true=y_test, y_pred=y_pred))

        # TRAINING LINEAR SVM
        # # C_range = 10.0 ** np.arange(-3, 3)
        # C_range = [0.1]
        # for i in range(len(C_range)):
        #     beta = 2
        #     t1 = time.time()
        #     # weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
        #     # cw = {0: weights[0], 1: weights[1]}
        #     linear_svm = SVC(kernel='linear', C=C_range[i], class_weight='balanced')
        #     linear_svm.fit(X_train, y_train)
        #
        #     print('-------------------------')
        #     print("C: ", C_range[i])
        #     print('training took ' + str(time.time() - t1) + ' seconds')
        #     y_pred = linear_svm.predict(X_test)
        #     precision = metrics.precision_score(y_true=y_test, y_pred=y_pred)
        #     recall = metrics.recall_score(y_true=y_test, y_pred=y_pred)
        #     f1 = 2 * precision * recall / (precision + recall)
        #     fbeta = (1+beta**2)*precision*recall/(beta**2 * precision + recall)
        #     print("linear svm accuracy ", metrics.accuracy_score(y_true=y_test, y_pred=y_pred))
        #     print("linear svm precision ", precision)
        #     print("linear recall ", recall)
        #     print("linear svm fbeta score", fbeta)


        # TRAINING RBF SVM
        C_range = [100]
        gamma_range = np.logspace(-2, 1, 10)
        pool = multiprocessing.Pool(processes=10)
        metricsQ=Queue()
        for i in range(len(C_range)):
            rbf_parallel=partial(j, i=i, metricsQueue=metricsQ, f=f, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, C_range=C_range, gamma_range=gamma_range)
            p=pool.map(rbf_parallel, np.arange(len(gamma_range)))
            p.start()
    
        print metrics.get()


                # if test_accuracy > best_accuracy:
                #     best_accuracy = test_accuracy
                # if fbeta > best_fbeta:
                #     best_fbeta = fbeta
                # if recall > best_recall:
                #     best_recall = recall
                # print('===========================', '\n')
    print("best accuracy", best_accuracy)
    print("best fbeta", best_fbeta)
    print("best recall", best_recall)


if __name__ == '__main__':
    main()
