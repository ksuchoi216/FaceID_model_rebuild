import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def show_pca(x, y_true):
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
        palette="deep"
    )
