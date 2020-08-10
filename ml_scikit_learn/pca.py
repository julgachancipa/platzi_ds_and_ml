import pandas as pd
import sklearn
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    dt_heart = pd.read_csv('/home/juliana/PycharmProjects/platzi_ds_and_ml/ml_scikit_learn/data/heart.csv')
    #print(dt_heart.head(5))

    dt_features = dt_heart.drop(['target'], axis=1)
    dt_target = dt_heart['target']

    dt_features = StandardScaler().fit_transform(dt_features)

    X_train, X_test, Y_train, Y_test = train_test_split(dt_features, dt_target, test_size=0.3, random_state=42)

    #print(X_train.shape)
    #print(Y_train.shape)

    pca = PCA(n_components=3)
    pca.fit(X_train)

    ipca = IncrementalPCA(n_components=3, batch_size=10)
    ipca.fit(X_train)

    plt.plot(range(len(pca.explained_variance_)), pca.explained_variance_ratio_)
    # plt.show()

    logistic = LogisticRegression(solver='lbfgs')

    dt_train = pca.transform(X_train)
    dt_test = pca.transform(X_test)
    logistic.fit(dt_train, Y_train)
    print('SCORE PCA:', logistic.score(dt_test, Y_test))

    dt_train = ipca.transform(X_train)
    dt_test = ipca.transform(X_test)
    logistic.fit(dt_train, Y_train)
    print('SCORE PCA:', logistic.score(dt_test, Y_test))
