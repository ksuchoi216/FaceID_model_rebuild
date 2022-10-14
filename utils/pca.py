import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def show_pca(x: np.ndarray, y_true: np.ndarray):
    if not isinstance(x, np.ndarray) or isinstance(y_true, np.ndarray):
        x = np.array(x)
        y_true = np.array(y_true)

    # standarize features
    scaler = StandardScaler()
    scaler.fit(x)
    standarized_x = scaler.transform(x)

    pca = PCA(n_components=2)
    pca.fit(standarized_x)

    pca_x = pca.transform(standarized_x)
    fig = plt.figure(1, figsize=(20, 10))

    pca_x1 = pca_x[:, 0]
    pca_x2 = pca_x[:, 1]

    data = np.column_stack((pca_x1, pca_x2, y_true))

    df = pd.DataFrame(data=data,
                      columns=["pca_x1", "pca_x2", "y_true"])

    df = df.astype({"y_true": int})
    print(f'df shape: {df.shape}')

    sns.scatterplot(
        data=df,
        x='pca_x1', y='pca_x2',
        hue="y_true",
        style='y_true',
        palette="deep",

    )
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
