# ML Spring 2018 Toxic Comments
# Sasha, Dalton, Danny, Amelia
# phase2.py -> better features and training the algorithm using TF-IDF and subset features


from kernel import *



def main():
    #Get our data
    X, y = util.get_data('../data/subsample_data.csv')
    print "Shapes are: ", X.shape, y.shape


    # We apply PCA and then train a dummy classifier
    X_small = applyPCA(X, 684)
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


    #Make our splits
    skf = StratifiedKFold(n_splits=5)

    # Starting RBF Params: Find optimal hyperparameters
    scoreCGvalue = {}
    # Loop through metrics to find optimal C and gamma values for each specific metric
    for metric in metric_list:
        scoreCGvalue[metric] = list(select_param_rbf(X_trainS, y_trainS, skf, metric=metric))
    print "C and Gamma values for PCA: ", scoreCGvalue

    #Lets go through the metrics again? This is efficient.
    for metricNDX in range(len(metric_list)):

        C, gamma = scoreCGvalue[metric_list[metricNDX]]
        print "Training with C: ", C, "and gamma: ", gamma

        #Train a model with its optimal c and gamma values (Currently only RBF)
        svmRBF = SVC(kernel='rbf', C=C, gamma=gamma)
        svmRBF.fit(X_trainS, y_trainS)

        #Let's see how we did!
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
