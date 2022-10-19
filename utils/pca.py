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
):
    if not isinstance(X, np.ndarray) or isinstance(y_true, np.ndarray):
        X = np.array(X)
        y_true = np.array(y_true)

    # standarize features
    scaler = StandardScaler()
    scaler.fit(X)
    X_standardized = scaler.transform(X)

    if isLDA:
        title = 'LDA'
        lda = LinearDiscriminantAnalysis(n_components=2)
        X_reduced = lda.fit_transform(X_standardized, y_true)
    else:
        title = 'PCA'
        pca = PCA(n_components=2)
        X_reduced = pca.fit_transform(X_standardized)

    # print(lda.coef_.shape)

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


def show_dimension_reduction_lda(
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    X_applied: np.ndarray,
    y_applied: np.ndarray

):
    print(f'fit: {X_fit.shape}, {y_fit.shape}')
    print(f'applied: {X_applied.shape}, {y_applied.shape}')

    scaler = StandardScaler()
    scaler.fit(X_fit)
    X_standardized = scaler.transform(X_fit)

    title = 'LDA'
    lda = LinearDiscriminantAnalysis(n_components=2)
    lda.fit(X_standardized, y_fit)

    X_standardized = scaler.transform(X_applied)
    X_reduced = lda.transform(X_standardized)

    fig = plt.figure(1, figsize=(20, 10))

    X1 = X_reduced[:, 0]
    X2 = X_reduced[:, 1]

    data = np.column_stack((X1, X2, y_applied))

    df = pd.DataFrame(data=data,
                      columns=["X1", "X2", "y_applied"])

    df = df.astype({"y_applied": int})
    # print(f'df shape: {df.shape}')

    sns.scatterplot(
        data=df,
        x='X1', y='X2',
        hue="y_applied",
        style='y_applied',
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


def sampleNumpyForEachLabel(
    X: np.ndarray,
    y: np.ndarray,
    num_ratio: float,
):

    labels = set(y)
    X_new = []
    y_new = []
    for label in labels:
        # print('label:', label)
        class_mask = np.where(y == label, True, False)
        X_ = X[class_mask, :]
        y_ = y[class_mask]
        # print('X_:', X_.shape)

        num, _ = X_.shape

        idx = np.random.randint(num, size=int(num*num_ratio))
        # print('idx:', idx.shape)
        X_ = X_[idx, :]
        y_ = y_[idx]

        X_new.append(X_)
        y_new.append(y_)

    # print(X_new[0].shape)

    num, dim = X_new[0].shape

    X_res = np.empty([1, dim])
    y_res = np.empty([1])

    # print("X_res:", X_res.shape)
    # print("y_res", y_res.shape)
    for X_, y_ in zip(X_new, y_new):
        X_res = np.vstack((X_res, X_))
        y_res = np.hstack((y_res, y_))
        # print('X_res:', X_res.shape)
        # print('y_res:', y_res.shape)

    X_res = X_res[1:, :]
    y_res = y_res[1:]
    # print('X_res:', X_res.shape)
    # print('y_res:', y_res.shape)

    return X_res, y_res
