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

def performance(y_true, y_pred, metric="accuracy") :
    """
    Used from project 6 (Dalton's code)
    Calculates the performance metric based on the agreement between the
    true labels and the predicted labels.

    Parameters
    --------------------
        y_true -- numpy array of shape (n,), known labels
        y_pred -- numpy array of shape (n,), (continuous-valued) predictions
        metric -- string, option used to select the performance measure
                  options: 'accuracy', 'f1_score', 'auroc', 'precision',
                           'sensitivity', 'specificity'

    Returns
    --------------------
        score  -- float, performance score
    """
    # map continuous-valued predictions to binary labels
    #y_label = np.sign(y_pred)
    y_label = y_pred # map points of hyperplane to +1
    if metric == "accuracy":
        score = metrics.accuracy_score(y_true, y_label)
    if metric == "f1_score":
        score = metrics.f1_score(y_true, y_label)

    # print "Y_True: ", y_true
    # print "y_pred: ", y_label
    confusionMatrix = metrics.confusion_matrix(y_true, y_label)

    if metric == "auroc":
        score = metrics.roc_auc_score(y_true, y_pred)
    if metric == "precision":
        score = metrics.precision_score(y_true, y_label)
    if metric == "sensitivity":
        score = float(confusionMatrix[1,1])/(confusionMatrix[1,1] + confusionMatrix[1, 0])
    if metric == "specificity":
        score = float(confusionMatrix[0,0])/(confusionMatrix[0,0] + confusionMatrix[0, 1])

    return score

def cv_performance(clf, X, y, kf, metric="accuracy") :
    """
    Used from Project 6
    Splits the data, X and y, into k-folds and runs k-fold cross-validation.
    Trains classifier on k-1 folds and tests on the remaining fold.
    Calculates the k-fold cross-validation performance metric for classifier
    by averaging the performance across folds.

    Parameters
    --------------------
        clf    -- classifier (instance of SVC)
        X      -- numpy array of shape (n,d), feature vectors
                    n = number of examples
                    d = number of features
        y      -- numpy array of shape (n,), binary labels {1,-1}
        kf     -- model_selection.KFold or model_selection.StratifiedKFold
        metric -- string, option used to select performance measure

    Returns
    --------------------
        score   -- float, average cross-validation performance across k folds
    """

    scores = []
    for train, test in kf.split(X, y) :
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
        clf.fit(X_train, y_train)
        # use SVC.decision_function to make ``continuous-valued'' predictions
        y_pred = clf.predict(X_test)
        score = performance(y_test, y_pred, metric)
        if not np.isnan(score) :
            scores.append(score)
    return np.array(scores).mean()

def select_param_rbf(X, y, kf, metric="accuracy") :
    """
    From Project 6 (Dalton's code)
    Sweeps different settings for the hyperparameters of an RBF-kernel SVM,
    calculating the k-fold CV performance for each setting, then selecting the
    hyperparameters that 'maximize' the average k-fold CV performance.

    Parameters
    --------------------
        X       -- numpy array of shape (n,d), feature vectors
                     n = number of examples
                     d = number of features
        y       -- numpy array of shape (n,), binary labels {1,-1}
        kf      -- model_selection.KFold or model_selection.StratifiedKFold
        metric  -- string, option used to select performance measure

    Returns
    --------------------
        C        -- float, optimal parameter value for an RBF-kernel SVM
        gamma    -- float, optimal parameter value for an RBF-kernel SVM
    """

    print 'RBF SVM Hyperparameter Selection based on ' + str(metric) + ':'

    C_range = 2 ** np.arange(-5.0, 10)
    gamma_range = 2 ** np.arange(-10.0, 5)

    #create the grid
    mgrid = np.mgrid[0:15.0, 0:15.0]
    grid = mgrid[0]
    cIndex = 0

    #iterate through c values and gamma values
    for c in C_range:
        gIndex = 0
        for gamma in gamma_range:
            #make the svm
            svm = SVC(kernel="rbf", C=c, gamma=gamma)
            accuracy = cv_performance(svm, X, y, kf, metric)

            #Place it in the grid
            grid[gIndex][cIndex] = accuracy
            gIndex += 1
        cIndex += 1

    #find the best values of each column
    cList = np.argmax(grid, axis=1)
    gammaList = []
    g = 0
    #then find the highest scores for each column
    for v in cList:
        gammaList.append(grid[g][v])
        g += 1
    #find the proper index and use that to figure out the gamma and c values used
    gIndex = np.argmax(gammaList)
    cIndex = cList[gIndex]
    bestGamma = 2 ** (gIndex - 10)
    bestC = 2 ** (cIndex - 5)

    return bestC, bestGamma

def PCA(X) :
    """
    Perform Principal Component Analysis.
    This version uses SVD for better numerical performance when d >> n.

    Note that because the covariance matrix Sigma = XX^T is positive semi-definite,
    all its eigenvalues are non-negative.  So we can sort by l rather than |l|.

    Parameters
    --------------------
        X      -- numpy array of shape (n,d), features

    Returns
    --------------------
        U      -- numpy array of shape (d,d), d d-dimensional eigenvectors
                  each column is a unit eigenvector; columns are sorted by eigenvalue
        mu     -- numpy array of shape (d,), mean of input data X
    """
    n, d = X.shape
    mu = np.mean(X, axis=0)
    x, l, v = np.linalg.svd(X-mu)
    l = np.hstack([l, np.zeros(v.shape[0] - l.shape[0], dtype=float)])
    U = np.array([vi/1.0 \
                  for (li, vi) \
                  in sorted(zip(l, v), reverse=True, key=lambda x: x[0])]).T
    return U, mu


def apply_PCA_from_Eig(X, U, l, mu) :
    """
    Project features into lower-dimensional space.

    Parameters
    --------------------
        X  -- numpy array of shape (n,d), n d-dimensional features
        U  -- numpy array of shape (d,d), d d-dimensional eigenvectors
              each column is a unit eigenvector; columns are sorted by eigenvalue
        l  -- int, number of principal components to retain
        mu -- numpy array of shape (d,), mean of input data X

    Returns
    --------------------
        Z   -- numpy matrix of shape (n,l), n l-dimensional features
               each row is a sample, each column is one dimension of the sample
        Ul  -- numpy matrix of shape (d,l), l d-dimensional eigenvectors
               each column is a unit eigenvector; columns are sorted by eigenvalue
               (Ul is a subset of U, specifically the d-dimensional eigenvectors
                of U corresponding to largest l eigenvalues)
    """
    n, d = X.shape
    Ul = U[:, :l]
    # Take l amount of columns from U, since it is sorted by eigenvalues just take l first (or last) columns

    # take X, U, UL and mu to calculate Z

    print "Shape of Ul: ", Ul.shape
    Z = np.dot((X-mu), Ul)

    return Z, Ul


def applyPCA(X, l):
    """ this applies the PCA reduction and returns the data with l dimensions
    """
    U, mu = PCA(X)
    Z, Ul = apply_PCA_from_Eig(X, U, l, mu)
    return Z


def main():
    #Get our data
    X, y = util.get_data('../data/subsample_data.csv')
    print "Shapes are: ", X.shape, y.shape

    X_small = applyPCA(X, 500)
    metric_list = ["accuracy", "f1_score", "precision", "sensitivity", "specificity"]
    #Using PCA:
    X_trainS, X_testS, y_trainS, y_testS = train_test_split(X_small, y, test_size=0.2)
    baseline = DummyClassifier(strategy='most_frequent')
    baseline.fit(X_trainS, y_trainS)
    print "Baseline Metrics: ", metrics.accuracy_score(baseline.predict(X_testS), y_testS)
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
        scoreCGvalue[metric] = list(select_param_rbf(X_trainS, y_trainS, skf, metric=metric))
    print "C and Gamma values for PCA: ", scoreCGvalue
    for metricNDX in range(len(metric_list)):
        C, gamma = scoreCGvalue[metric_list[metricNDX]]
        print "Training with C: ", C, "and gamma: ", gamma
        svmRBF = SVC(kernel='rbf', C=C, gamma=gamma)
        svmRBF.fit(X_trainS, y_trainS)
        print "METRIC IS: ", metric_list[metricNDX]
        print "Baseline Performance: ", performance(y_testS, baseline.predict(X_testS), metric=metric_list[metricNDX])
        print "SVM Performance is: ", performance(y_testS, svmRBF.predict(X_testS), metric=metric_list[metricNDX])




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
