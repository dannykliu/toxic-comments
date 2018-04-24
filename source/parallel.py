from __future__ import print_function

import util
import numpy as np

# scikit-learn libraries
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import time
import multiprocessing
from functools import partial


def get_metrics(y_true, y_pred):
    beta = 1.5
    confusion_matrix = metrics.confusion_matrix(y_true=y_pred, y_pred=y_pred)
    if confusion_matrix[1, 1] + confusion_matrix[1, 0] == 0:
        recall = float('NaN')
    else:
        recall = float(confusion_matrix[1, 1]) / (confusion_matrix[1, 1] + confusion_matrix[1, 0])
    if confusion_matrix[1, 1] + confusion_matrix[0, 1] == 0:
        precision = float('NaN')
    else:
        precision = float(confusion_matrix[1, 1]) / (confusion_matrix[1, 1] + confusion_matrix[0, 1])
    if beta ** 2 * precision + recall == 0:
        fbeta = float('NaN')
    else:
        fbeta = (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall)
    accuracy = float(np.sum(y_true == y_pred)) / len(y_true)
    return accuracy, precision, recall, fbeta


def report_cv_performance(name, clf, X, y, kf, C, gamma):
    if C and gamma:
        print("C: ", C, "gamma: ", gamma)
    accuracies, precisions, recalls, fbetas = [], [], [], []
    for train, test in kf.split(X, y):
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy, precision, recall, fbeta = get_metrics(y_true=y_test, y_pred=y_pred)
        if not np.isnan(accuracy):
            accuracies.append(accuracy)
        if not np.isnan(precision):
            precisions.append(precision)
        if not np.isnan(recall):
            recalls.append(recall)
        if not np.isnan(fbeta):
            fbetas.append(fbeta)
    print("--------------------------")
    print(name, "cv accuracy", np.mean(accuracies))
    print(name, "cv precision", np.mean(precisions))
    print(name, "cv recall", np.mean(recalls))
    print(name, "cv f1.5 score", np.mean(fbetas))


def parallel_rbf(j, i, X, y, kf, C_range, gamma_range):
    clf = SVC(kernel='rbf', C=C_range[i], gamma=gamma_range[j], class_weight='balanced')
    report_cv_performance('rbf', clf, X=X, y=y, kf=kf, C=C_range[i], gamma=gamma_range[j])


def tune_rbf(X, y):
    skf = StratifiedKFold(n_splits=5)
    C_range = [0.01, 0.1, 1.0, 10.0, 100.0]
    gamma_range = np.logspace(-2, 1, 10)
    pool = multiprocessing.Pool(processes=10)
    for i in range(len(C_range)):
        rbf_parallel = partial(parallel_rbf, i=i, X=X, y=y, kf=skf, C_range=C_range, gamma_range=gamma_range)
        p = pool.map(rbf_parallel, range(len(gamma_range)))
        # p.start()
    p.join()


def main():
    t1 = time.time()
    X, y, raw = util.get_data('../data/subset.csv')
    new_features = util.get_features(raw)  # get homegrown features
    vect = TfidfVectorizer(min_df=2)
    X_dtm = vect.fit_transform(X)
    info_gains = np.apply_along_axis(util.info_gain, 0, X_dtm.toarray(), y, 0.00001)
    num_features = 2000
    max_cols = info_gains.argsort()[-num_features:][::-1]
    # print_vocab(vect, max_cols)
    X = X_dtm[:, max_cols].toarray()  # turn X from sparse matrix to numpy array
    for new_feature in new_features:  # add our features as columns to X
        X = np.append(X, new_feature.reshape(-1, 1), axis=1)
    print("data matrix shape", X.shape)
    print("preprocessing took", str(time.time() - t1), "seconds")
    tune_rbf(X, y)


if __name__ == '__main__':
    main()
