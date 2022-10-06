import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
<<<<<<< HEAD
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
    
=======


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
>>>>>>> 82c37636f5892184c7f1117fb6c4af1ff064877b
