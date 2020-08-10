import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

if __name__ == "__main__":

    dt_heart = pd.read_csv('/home/juliana/PycharmProjects/platzi_ds_and_ml/ml_scikit_learn/data/heart.csv')
    print(dt_heart['target'].describe())

    X = dt_heart.drop(['target'], axis=1)
    Y = dt_heart['target']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.35)

    boost = GradientBoostingClassifier(n_estimators=50).fit(X_train, Y_train)
    boost_pred = boost.predict(X_test)
    print("=" * 64)
    print('Accuracy Boosting: ', accuracy_score(boost_pred, Y_test))
