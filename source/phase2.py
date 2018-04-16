# ML Spring 2018 Toxic Comments
# Sasha, Dalton, Danny, Amelia
# phase2.py -> better features and training the algorithm using TF-IDF and subset features


from kernel import *
import nltk
import multiprocessing as mp
from sklearn.feature_extraction.text import TfidfVectorizer


def train(inputList):
    [X_train, y_train, metric] = inputList
    skf = StratifiedKFold(n_splits=5)
    metric= list(select_param_rbf(X_train, y_train, skf, metric=metric, class_weight='balanced'))
    return metric

def main():
    #Get our data
    #USE GETDATA2 NOW
    file = open("SVMRBFResults.txt", "w")
    X, y = util.get_data2('../data/subset.csv')
    file.write("Shapes are: "+ str(X.shape)+ str(y.shape))


    vect = TfidfVectorizer(max_features=5000, min_df=2)
    X_dtm = vect.fit_transform(X)
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
    inputs = []
    for metric in metric_list:
        input = [X_train, y_train, metric]
        inputs.append(input)
    # Starting RBF Params: Find optimal hyperparameters
    scoreCGvalue = {}
    # Loop through metrics to find optimal C and gamma values for each specific metric
    pool = mp.Pool(5)
    outputs = pool.map(train, inputs)

    file.write ("THESE ARE THE OUTPUTS: "+ str(outputs)+"\n")
    file.write ("C and Gamma values Training: "+ str(scoreCGvalue)+"\n")

    scoreCGvalue = outputs
    #Lets go through the metrics again? This is efficient.
    for metricNDX in range(len(metric_list)):

        C, gamma = scoreCGvalue[metric_list[metricNDX]]
        file.write ("Training with C: "+ C+ "and gamma: "+ gamma +"\n")

        #Train a model with its optimal c and gamma values (Currently only RBF)
        svmRBF = SVC(kernel='rbf', C=C, gamma=gamma, class_weight= 'balanced')
        svmRBF.fit(X_train, y_train)

        #Let's see how we did!
        file.write ("METRIC IS: "+ str(metric_list[metricNDX])+"\n")
        file.write ("Baseline Performance: "+ str(performance(y_test, baseline.predict(X_test), metric=metric_list[metricNDX]))+"\n")
        file.write ("SVM Performance is: "+ str(performance(y_test, svmRBF.predict(X_test), metric=metric_list[metricNDX])) +"\n")
        file.flush()

    file.close()

if __name__ == '__main__':
    main()
