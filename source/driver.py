from __future__ import print_function

import util
import numpy as np

# scikit-learn libraries
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import StratifiedKFold


def print_vocab(vect, max_cols):
    # printing vocab
    # vocab = vect.vocabulary_
    # inv_map = {v: k for k, v in vocab.iteritems()}
    # for i in range(len(max_cols)):
    #     print(inv_map[max_cols[i]], end=' ')
    pass


def plot_tfidf(vect, X, y):
    # # plotting for tfidf threshold
    # zeros = np.where(y == 0)[0]
    # ones = np.where(y == 1)[0]
    # response = vect.transform(X[zeros])
    # a = response.toarray().flatten()[::-1]
    # plt.hist(a,  np.arange(0.00001, 1.01, 0.01))
    # plt.show()
    pass


def get_metrics(y_true, y_pred):
    beta = 1.5
    confusion_matrix = metrics.confusion_matrix(y_true=y_true, y_pred=y_pred)
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


def report_cv_performance(name, clf, X, y, kf):
    t1 = time.time()
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
    print("cross validation took", str(time.time()-t1), "seconds")
    print(name, "cv accuracy", np.mean(accuracies))
    print(name, "cv precision", np.mean(precisions))
    print(name, "cv recall", np.mean(recalls))
    print(name, "cv f1.5 score", np.mean(fbetas))


def train_baseline(X_train, X_test, y_train, y_test):
    baseline = DummyClassifier(strategy='stratified')
    baseline.fit(X_train, y_train)
    y_pred = baseline.predict(X_test)
    print("baseline accuracy ", metrics.accuracy_score(y_true=y_test, y_pred=y_pred))
    print("baseline precision ", metrics.precision_score(y_true=y_test, y_pred=y_pred))
    print("baseline recall ", metrics.recall_score(y_true=y_test, y_pred=y_pred))


def tune_linear(X, y):
    skf = StratifiedKFold(n_splits=5)
    C_range = [0.01, 0.1, 1.0, 10.0, 100.0]
    for i in range(len(C_range)):
        print("--------------------------")
        print("C:", C_range[i])
        clf = SVC(kernel='linear', C=C_range[i], class_weight='balanced')
        report_cv_performance('rbf', clf, X=X, y=y, kf=skf)


def tune_rbf(X, y):
    skf = StratifiedKFold(n_splits=5)
    C_range = [0.01, 0.1, 1.0, 10.0, 100.0]
    gamma_range = np.logspace(-2, 1, 10)
    for i in range(len(C_range)):
        for j in range(len(gamma_range)):
            print("--------------------------")
            print("C:", C_range[i], "gamma:", gamma_range[j])
            clf = SVC(kernel='rbf', C=C_range[i], gamma=gamma_range[j], class_weight='balanced', cache_size=2000)
            report_cv_performance('rbf', clf, X=X, y=y, kf=skf)


def tune_dt(X, y):
    skf = StratifiedKFold(n_splits=5)
    depths = [80, 90, 100, 110, 120, 130, 140, 150, 160]
    for i in range(len(depths)):
        print("--------------------------")
        print("Depth:", depths[i])
        clf = DecisionTreeClassifier(max_depth=depths[i], class_weight='balanced')
        report_cv_performance('dt', clf, X=X, y=y, kf=skf)


def tune_rf(X, y):
    skf = StratifiedKFold(n_splits=5)
    depths = [80, 90, 100, 110]
    estimators = [10, 20, 30]
    for i in range(len(depths)):
        for j in range(len(estimators)):
            print("--------------------------")
            print("Max Depth:", depths[i], "Num Estimators:", estimators[j])
            clf = RandomForestClassifier(max_depth=depths[i], n_estimators=estimators[j], class_weight='balanced')
            report_cv_performance('rf', clf, X=X, y=y, kf=skf)


def main():
    t1 = time.time()
    X, y, raw = util.get_data('../data/subset.csv')
    new_features = util.get_features(raw)  # get homegrown features
    vect = TfidfVectorizer(min_df=2)
    X_dtm = vect.fit_transform(X)
    info_gains = np.apply_along_axis(util.info_gain, 0, X_dtm.toarray(), y, 0.00001)
    num_features = 500
    max_cols = info_gains.argsort()[-num_features:][::-1]
    # print_vocab(vect, max_cols)
    X = X_dtm[:, max_cols].toarray()  # turn X from sparse matrix to numpy array
    for new_feature in new_features:  # add our features as columns to X
        X = np.append(X, new_feature.reshape(-1, 1), axis=1)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    print(X.shape)
    print("preprocessing took", str(time.time() - t1), "seconds")

    # train_baseline(X_train, X_test, y_train, y_test)
    # tune_linear(X_train, X_test, y_train, y_test)
    # tune_dt(X_train, X_test, y_train, y_test)
    tune_rbf(X, y)
    # tune_rf(X_train, X_test, y_train, y_test)


if __name__ == '__main__':
    main()
