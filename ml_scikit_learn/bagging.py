import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

if __name__ == "__main__":

    dt_heart = pd.read_csv('/home/juliana/PycharmProjects/platzi_ds_and_ml/ml_scikit_learn/data/heart.csv')
    print(dt_heart['target'].describe())

    X = dt_heart.drop(['target'], axis=1)
    Y = dt_heart['target']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.35)

    knn_class = KNeighborsClassifier().fit(X_train, Y_train)
    knn_pred = knn_class.predict(X_test)

    print('=' * 64)
    print('Accuracy KNeighbors: ', accuracy_score(knn_pred, Y_test))

    bag_class = BaggingClassifier(base_estimator=KNeighborsClassifier(), n_estimators=5).fit(X_train, Y_train)
    bag_pred = bag_class.predict(X_test)

    print('=' * 64)
    print('Accuracy Bagging: ', accuracy_score(bag_pred, Y_test))
