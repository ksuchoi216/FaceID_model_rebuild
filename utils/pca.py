import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def execute_pca(x, y_true):
    sns.set(rc={'figure.figsize': (16, 10)})

    if not isinstance(x, np.ndarray) or not isinstance(y_true, np.ndarray):
        print(f'Please convert input data type to numpy array!!')
        raise TypeError

    print(x.shape, y_true.shape)

    scaler = StandardScaler()
    scaler.fit(x)
    standarized_data = scaler.transform(x)

    pca = PCA(n_components=2)
    pca.fit(standarized_data)

    pca_x = pca.transform(standarized_data)
    # fig = plt.figure(1, figsize=(20, 10))
    pca_x1 = pca_x[:, 0]
    pca_x2 = pca_x[:, 1]

    data = np.column_stack((pca_x1, pca_x2, y_true))

    df = pd.DataFrame(data=data, columns=['pca_x1', 'pca_x2', 'y_true'])
    df.astype({'y_true': 'int8'})

    df.info

    sns.scatterplot(data=df, x='pca_x1', y='pca_x2',
                    hue='y_true', style='y_true', palette='deep')
