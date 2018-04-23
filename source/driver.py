from __future__ import print_function

import util
import numpy as np

# scikit-learn libraries
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import time


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


def get_metrics(y_test, y_test_pred, y_train, y_train_pred):
    beta = 2.0
    confusion_matrix = metrics.confusion_matrix(y_true=y_test, y_pred=y_test_pred)
    if confusion_matrix[0, 0] + confusion_matrix[0, 1] == 0:
        specificity = 0
    else:
        specificity = float(confusion_matrix[0, 0]) / (confusion_matrix[0, 0] + confusion_matrix[0, 1])
    if confusion_matrix[1, 1] + confusion_matrix[1, 0] == 0:
        recall = 0
    else:
        recall = float(confusion_matrix[1, 1]) / (confusion_matrix[1, 1] + confusion_matrix[1, 0])
    if confusion_matrix[1, 1] + confusion_matrix[0, 1] == 0:
        precision = 0
    else:
        precision = float(confusion_matrix[1, 1]) / (confusion_matrix[1, 1] + confusion_matrix[0, 1])
    if beta ** 2 * precision + recall == 0:
        fbeta = 0
    else:
        fbeta = (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall)
    test_accuracy = float(np.sum(y_test == y_test_pred)) / len(y_train)
    train_accuracy = float(np.sum(y_train == y_train_pred)) / len(y_train)
    return test_accuracy, train_accuracy, fbeta, precision, recall, specificity


def report_metrics(name, clf, X_train, X_test, y_train, y_test, best_accuracy, best_fbeta, best_recall, best_specificity):
    t1 = time.time()
    y_test_pred = clf.predict(X_test)
    y_train_pred = clf.predict(X_train)
    test_accuracy, train_accuracy, fbeta, precision, recall, specificity = get_metrics(y_test, y_test_pred, y_train, y_train_pred)
    print(name, "train accuracy", train_accuracy)
    print(name, "test accuracy", test_accuracy)
    print(name, "test precision", precision)
    print(name, "test recall", recall)
    print(name, "test specificity", specificity)
    print(name, "test f2 score", fbeta)
    print("metrics and predictions took " + str(time.time() - t1) + ' seconds')
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
    if fbeta > best_fbeta:
        best_fbeta = fbeta
    if recall > best_recall:
        best_recall = recall
    if specificity > best_specificity:
        best_specificity = specificity
    return best_accuracy, best_fbeta, best_recall, best_specificity


def train_baseline(X_train, X_test, y_train, y_test):
    baseline = DummyClassifier(strategy='stratified')
    baseline.fit(X_train, y_train)
    y_pred = baseline.predict(X_test)
    print("baseline accuracy ", metrics.accuracy_score(y_true=y_test, y_pred=y_pred))
    print("baseline precision ", metrics.precision_score(y_true=y_test, y_pred=y_pred))
    print("baseline recall ", metrics.recall_score(y_true=y_test, y_pred=y_pred))


def train_linear(X_train, X_test, y_train, y_test):
    best_accuracy, best_fbeta, best_recall, best_specificity = 0, 0, 0, 0
    C_range = [0.01, 0.1, 1.0, 10.0, 100.0]  # 0.1
    for i in range(len(C_range)):
        t1 = time.time()
        linear_svm = SVC(kernel='linear', C=C_range[i], class_weight='balanced')
        linear_svm.fit(X_train, y_train)
        print('-------------------------')
        print("C: ", C_range[i])
        print('training took ' + str(time.time() - t1) + ' seconds')
        best_accuracy, best_fbeta, best_recall, best_specificity = report_metrics('linear', linear_svm, X_train, X_test, y_train, y_test, best_accuracy, best_fbeta, best_recall, best_specificity)

    print("\nbest test accuracy", best_accuracy)
    print("best f2", best_fbeta)
    print("best recall", best_recall)
    print("best specificity", best_specificity)


def train_rbf(X_train, X_test, y_train, y_test):
    best_accuracy, best_fbeta, best_recall, best_specificity = 0, 0, 0, 0
    C_range = [0.01, 0.1, 1.0, 10.0, 100.0]
    gamma_range = np.logspace(-2, 1, 10)
    for i in range(len(C_range)):
        for j in range(len(gamma_range)):
            t1 = time.time()
            rbf_svm = SVC(kernel='rbf', C=C_range[i], gamma=gamma_range[j], class_weight='balanced')
            rbf_svm.fit(X_train, y_train)
            print('---------------------------------')
            print("C: ", C_range[i], "gamma: ", gamma_range[j])
            print('training took ' + str(time.time() - t1) + ' seconds')
            best_accuracy, best_fbeta, best_recall, best_specificity = report_metrics('rbf', rbf_svm, X_train, X_test, y_train, y_test, best_accuracy, best_fbeta, best_recall, best_specificity)

    print("\nbest test accuracy", best_accuracy)
    print("best f2", best_fbeta)
    print("best recall", best_recall)
    print("best specificity", best_specificity)


def train_rf(X_train, X_test, y_train, y_test):
    best_accuracy, best_fbeta, best_recall, best_specificity = 0, 0, 0, 0
    # depths = np.arange(40, 80, 4)
    # estimators = np.arange(20, 50, 4)
    depths = [80, 90, 100, 110]
    estimators = [10, 20, 30]
    min_impurity_decrease = [0]  # np.arange(0, 0.0002, 0.00002)
    min_samples_leaf = np.arange(1, 10, 1)
    for i in range(len(depths)):
        for j in range(len(estimators)):
            for k in range(len(min_impurity_decrease)):
                for l in range(len(min_samples_leaf)):
                    t1 = time.time()
                    rf = RandomForestClassifier(max_depth=depths[i], n_estimators=estimators[j], min_impurity_decrease=min_impurity_decrease[k], min_samples_leaf=min_samples_leaf[l], class_weight='balanced_subsample')
                    rf.fit(X_train, y_train)
                    print('---------------------------------')
                    print("Depth: ", depths[i], "Estimators: ", estimators[j], "Impurity Decrease: ", min_impurity_decrease[k], 'Min Samples leaf: ', min_samples_leaf[l])
                    print('training took ' + str(time.time() - t1) + ' seconds')
                    best_accuracy, best_fbeta, best_recall, best_specificity = report_metrics('rf', rf, X_train, X_test, y_train, y_test, best_accuracy, best_fbeta, best_recall, best_specificity)

    print("\nrf best test accuracy", best_accuracy)
    print("rf best f2", best_fbeta)
    print("rf best recall", best_recall)
    print("rf best specificity", best_specificity)


def main():
    t1 = time.time()
    X, y, raw = util.get_data('../data/subset.csv')
    new_features = util.get_features(raw)  # get homegrown features
    vect = TfidfVectorizer(min_df=2)
    X_dtm = vect.fit_transform(X)
    info_gains = np.apply_along_axis(util.info_gain, 0, X_dtm.toarray(), y, 0.00001)
    num_features = 900
    max_cols = info_gains.argsort()[-num_features:][::-1]
    # print_vocab(vect, max_cols)
    X = X_dtm[:, max_cols].toarray()  # turn X from sparse matrix to numpy array
    for new_feature in new_features:  # add our features as columns to X
        X = np.append(X, new_feature.reshape(-1, 1), axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
    print(X_train.shape, X_test.shape)
    print("preprocessing took", str(time.time() - t1), "seconds")
    # train_baseline(X_train, X_test, y_train, y_test)
    # train_linear(X_train, X_test, y_train, y_test)
    # train_rbf(X_train, X_test, y_train, y_test)
    train_rf(X_train, X_test, y_train, y_test)


if __name__ == '__main__':
    main()
