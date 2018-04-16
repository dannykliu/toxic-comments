# ML Spring 2018 Toxic Comments
# Sasha, Dalton, Danny, Amelia

import util
import numpy as np

# matplotlib libraries
import matplotlib.pyplot as plt

# scikit-learn libraries
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from kernel import *


def main():
    #Get our data
    X, y = util.get_data('../data/subsample_data.csv')
    print "Shapes are: ", X.shape, y.shape


    metric_list = ["accuracy", "f1_score", "precision", "sensitivity", "specificity"]
    #Using PCA:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    baseline = DummyClassifier(strategy='most_frequent')
    baseline.fit(X_train, y_train)
    print "Baseline Metrics: ", metrics.accuracy_score(baseline.predict(X_test), y_test)
    # linear_svm = SVC(kernel='linear', C=1.0)
    # linear_svm.fit(X_trainS, y_trainS)
    #
    # for metricNDX in range(len(metric_list)):
    #     print "METRIC IS: ", metric_list[metricNDX]
    #     if metric_list[metricNDX] == "auroc":
    #         print "Omitting Baseline due to metric"
    #         print "Omitting SVM due to metric"
    #         #print "SVM Performance is: ", performance(y_testS, linear_svm.decision_function(X_testS), metric=metric_list[metricNDX])
    #     print "Baseline Performance: ", performance(y_testS, baseline.predict(X_testS), metric=metric_list[metricNDX])
    #     print "SVM Performance is: ", performance(y_testS, linear_svm.predict(X_testS),metric=metric_list[metricNDX])

    skf = StratifiedKFold(n_splits=5)

    #Find optimal hyperparameters
    scoreCGvalue = {}
    for metric in metric_list:
        scoreCGvalue[metric] = list(select_param_rbf(X_train, y_train, skf, metric=metric))
    print "C and Gamma values for PCA: ", scoreCGvalue
    for metricNDX in range(len(metric_list)):
        C, gamma = scoreCGvalue[metric_list[metricNDX]]
        print "Training with C: ", C, "and gamma: ", gamma
        svmRBF = SVC(kernel='rbf', C=C, gamma=gamma)
        svmRBF.fit(X_train, y_train)
        print "METRIC IS: ", metric_list[metricNDX]
        print "Baseline Performance: ", performance(y_test, baseline.predict(X_test), metric=metric_list[metricNDX])
        print "SVM Performance is: ", performance(y_test, svmRBF.predict(X_test), metric=metric_list[metricNDX])




    ### Regular testing (without PCA)
    #Baseline training
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # baseline = DummyClassifier(strategy='most_frequent')
    # baseline.fit(X_train, y_train)
    # print "Baseline Metrics: ", metrics.accuracy_score(baseline.predict(X_test), y_test)
    #
    # skf = StratifiedKFold(n_splits=5)
    # metric_list = ["accuracy", "f1_score", "auroc", "precision", "sensitivity", "specificity"]
    # #Find optimal hyperparameters
    # scoreCGvalue = {}
    # for metric in metric_list:
    #     scoreCGvalue[metric] = list(select_param_rbf(X_train, y_train, skf, metric=metric))
    # print "C and Gamma Values: ", scoreCGvalue


    #SVM training
    # svmRBF = SVC(kernel='rbf', C=1.0)
    # linear_svm.fit(X_train, y_train)
    # print 'done training'
    # print metrics.accuracy_score(linear_svm.predict(X_test), y_test)
    # print np.equal(linear_svm.predict(X_test), baseline.predict(X_test))


if __name__ == '__main__':
    main()
