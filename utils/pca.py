from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import pandas as pd
def execute_pca(numpy_data):
  scaler=StandardScaler()
  scaler.fit(numpy_data)
  standarized_data=scaler.transform(numpy_data)

  pca = PCA(n_components=2)
  pca.fit(standarized_data)

  pca_x=pca.transform(standarized_data)
  # fig = plt.figure(1, figsize=(20, 10))
  pca_x1 = pca_x[:, 0]
  pca_x2 = pca_x[:, 1]

  return pca_x1, pca_x2


def show_pca(numpy_data_x, numpy_data_y, label_list = None):
  
  train_x = numpy_data_x
  train_y = numpy_data_y
  
  #standarize features
  scaler=StandardScaler()
  scaler.fit(train_x)
  standarized_x=scaler.transform(train_x)

  pca = PCA(n_components=2)
  pca.fit(standarized_x)

  pca_x=pca.transform(standarized_x)
  fig = plt.figure(1, figsize=(20, 10))
  
  plot = plt.scatter(pca_x[:,0],pca_x[:,1],c=train_y)
  plt.plot()
  
  
  # plt.axis('off')
  plt.legend(handles = plot.legend_elements()[0], labels = label_list)
  plt.grid(True)
  plt.show()

import pandas as pd

def show_pca_with_prediction(x_emb, y_true, y_pred):
  #standarize features
  scaler=StandardScaler()
  scaler.fit(x_emb)
  standarized_x=scaler.transform(x_emb)

  pca = PCA(n_components=2)
  pca.fit(standarized_x)

  pca_x=pca.transform(standarized_x)
  fig = plt.figure(1, figsize=(20, 10))
  x1 = pca_x[:, 0]
  x2 = pca_x[:, 1]
  
  correction = np.where(y_true == y_pred, 1, 0)
  
  df = pd.DataFrame(data=[x1, x2, y_true, y_pred, correction], columns=["pca1", "pca2", "y_true", "y_pred", "correction" ]).T

  # sns.scatterplot(data = df, x=)  
  # plt.scatter(x1, x2, color='red', marker="x")
  
  # plt.axis('off')
  # plt.legend(handles = plot.legend_elements()[0], labels = label_list)
  plt.grid(True)
  plt.show()
  

  