import pandas as pd

from sklearn.cluster import MeanShift

if __name__ == "__main__":

    dataset = pd.read_csv("/home/juliana/PycharmProjects/platzi_ds_and_ml/ml_scikit_learn/data/candy.csv")
    print(dataset.head(5))

    X = dataset.drop('competitorname', axis=1)

    meanshift = MeanShift().fit(X)
    print(meanshift.labels_)
    print("="*64)
    print(meanshift.cluster_centers_)

    dataset['meanshift'] = meanshift.labels_
    print("="*64)
    print(dataset)
