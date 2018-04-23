# ML Spring 2018 Toxic Comments
# Sasha, Dalton, Danny, Amelia
# ensemble.py -> training ensemble methods!


from kernel import *
import nltk
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

def main():

    #Get our data
    #USE GETDATA2 NOW
    file = open("DecisionTreeTuningResults.txt", "w")
    X, y, raw = util.get_data('../data/subset.csv')
    new_features = util.get_features(raw)
    file.write("Shapes are: "+ str(X.shape)+ str(y.shape))


    vect = TfidfVectorizer(max_features=None, min_df=2)
    X_dtm = vect.fit_transform(X)
    info_gains = np.apply_along_axis(util.info_gain, 0, X_dtm.toarray(), y, 0.0001)
    max_cols = info_gains.argsort()[-2000:][::-1]


    X = X_dtm[:, max_cols].toarray()
    for feature in new_features:
        X = np.append(X, feature.reshape(-1,1), axis=1)
    print(X_dtm.shape, y.shape)

    metric_list = ["accuracy", "f1_score", "precision", "sensitivity", "specificity"]
    # still need to add in tf-idf implementation
    X_train, X_test, y_train, y_test = train_test_split(X_dtm, y, test_size=0.2)
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    baseline = DummyClassifier(strategy='uniform')
    baseline.fit(X_train, y_train)

    file.write ("Baseline Metrics: "+ str(metrics.accuracy_score(baseline.predict(X_test), y_test))+"\n")
    file.flush()
    #Make our splits
    # Starting ensemble methods

    # #Train a model with its optimal c and gamma values (Currently only RBF)
    # #ensemble = AdaBoostClassifier()
    # highestRecall = [0, 1, 0]
    # for estimator in range(5, 20):
    #     for depth in range(2, 15):
    #         ensemble = RandomForestClassifier(criterion= "entropy", class_weight="balanced", max_features= "auto", n_estimators = estimator, max_depth = depth )
    #         ensemble.fit(X_train, y_train)
    #         recall = performance(y_test, ensemble.predict(X_test), metric="sensitivity")
    #         if(recall > highestRecall[0]):
    #             highestRecall = [recall, estimator, depth]
    #         file.write ("Estimator is: "+ str(estimator) +"\n")
    #         file.write ("Max_depth is: "+ str(depth) +"\n")
    #         for metric in metric_list:
    #             #Let's see how we did!
    #             file.write ("METRIC IS: "+ str(metric)+"\n")
    #             file.write ("Baseline Performance: "+ str(performance(y_test, baseline.predict(X_test), metric=metric))+"\n")
    #             file.write ("Random Forest Performance is: "+ str(performance(y_test, ensemble.predict(X_test), metric=metric)) +"\n")
    #             file.flush()

    #Train a model with its optimal c and gamma values (Currently only RBF)
    #ensemble = AdaBoostClassifier()
    highestRecall = [0, 0, 0]
    for leaf in range()
    for depth in range(2, 15):
        ensemble = DecisionTreeClassifier(criterion= "entropy", class_weight="balanced", max_features= "auto", max_depth = depth )
        ensemble.fit(X_train, y_train)
        recall = performance(y_test, ensemble.predict(X_test), metric="sensitivity")
        if(recall > highestRecall[0]):
            highestRecall = [recall, depth]
        file.write ("Max_depth is: "+ str(depth) +"\n")
        for metric in metric_list:
            #Let's see how we did!
            file.write ("METRIC IS: "+ str(metric)+"\n")
            file.write ("Baseline Performance: "+ str(performance(y_test, baseline.predict(X_test), metric=metric))+"\n")
            file.write ("Decision Tree Performance is: "+ str(performance(y_test, ensemble.predict(X_test), metric=metric)) +"\n")
            file.flush()




    file.close()

if __name__ == '__main__':
    main()
