import util

# scikit-learn libraries
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


def main():
    X, y = util.get_subsampled_data()
    print X.shape, y.shape
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    baseline = DummyClassifier()
    baseline.fit(X_train, y_train)
    print metrics.accuracy_score(baseline.predict(X_test), y_test)
    linear_svm = SVC(kernel='linear', C=1.0)
    linear_svm.fit(X_train, y_train)
    print 'done training'
    print metrics.accuracy_score(linear_svm.predict(X_test), y_test)



if __name__ == '__main__':
    main()
