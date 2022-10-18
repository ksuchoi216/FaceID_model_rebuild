import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def show_dimension_reduction(
    X: np.ndarray,
    y_true: np.ndarray,
    isLDA=False,
    X_extra=None,
    y_extra=None
):
    if not isinstance(X, np.ndarray) or isinstance(y_true, np.ndarray):
        X = np.array(X)
        y_true = np.array(y_true)

    # standarize features
    scaler = StandardScaler()
    scaler.fit(X)
    X_standarized = scaler.transform(X)

    if isLDA:
        title = 'LDA'
        lda = LinearDiscriminantAnalysis(n_components=2)
        lda.fit(X_standarized, y_true)
        if X_extra is not None and y_extra is not None:
            X_reduced = lda.transform(X_extra)
        else:
            X_reduced = lda.transform(X_standarized)
    else:
        title = 'PCA'
        pca = PCA(n_components=2)
        X_reduced = pca.fit_transform(X_standarized)

    print(lda.coef_.shape)

    fig = plt.figure(1, figsize=(20, 10))

    X1 = X_reduced[:, 0]
    X2 = X_reduced[:, 1]

    data = np.column_stack((X1, X2, y_true))

    df = pd.DataFrame(data=data,
                      columns=["X1", "X2", "y_true"])

    df = df.astype({"y_true": int})
    print(f'df shape: {df.shape}')

    sns.scatterplot(
        data=df,
        x='X1', y='X2',
        hue="y_true",
        style='y_true',
        palette="deep",

    )
    plt.title(title)
    plt.show()


def show_screegraph(x: np.ndarray, n_components: int, info_percent=0.95):

    plt.figure(figsize=(14, 6), dpi=80)
    # standarize features
    scaler = StandardScaler()
    scaler.fit(x)
    standarized_x = scaler.transform(x)

    pca = PCA(n_components=n_components)
    pca.fit(standarized_x)

    projected_x = pca.transform(standarized_x)
    print(f'x: {standarized_x.shape} projected_x: {projected_x.shape}')

    plt.plot(pca.explained_variance_)
    plt.grid()
    plt.xlabel('Explained Variance')
    plt.figure()

    plt.figure(figsize=(14, 6), dpi=80)

    plt.plot(np.arange(len(pca.explained_variance_ratio_))+1,
             np.cumsum(pca.explained_variance_ratio_), 'o-')
    plt.axis([1, len(pca.explained_variance_ratio_), 0, 1])
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.title('Scree Graph')

    plt.grid()

    plt.show()
    percent = np.cumsum(pca.explained_variance_ratio_)
    print("The number of total dimension:", len(percent))
    the_number_of_dimension = np.where((percent >= info_percent))[0]
    print(
        f"The number of dimension to keep {info_percent}:\
            {the_number_of_dimension[0]}")
