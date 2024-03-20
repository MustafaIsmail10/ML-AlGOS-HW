import pickle
import numpy as np
import pickle
import matplotlib.pyplot as plt
import sklearn.datasets
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP


dataset = pickle.load(open("../data/part3_dataset.data", "rb"))


color_map = {0: "red", 1: "orange", 2: "green"}

# preprocessing dataset 1
mean1 = np.mean(dataset, axis=0)
std1 = np.std(dataset, axis=0)
preprocessed_dataset1 = (dataset - mean1) / std1


################# DATASET 1 #################

method = UMAP(
    n_neighbors=100, n_components=2
)  # can transform other data instances after training

projected_data = method.fit_transform(preprocessed_dataset1)
plt.scatter(projected_data[:, 0], projected_data[:, 1])
plt.show()
print(dataset.shape)


method = PCA(n_components=2)  # can transform other data instances after training
projected_data = method.fit_transform(preprocessed_dataset1)
plt.scatter(projected_data[:, 0], projected_data[:, 1])
plt.show()
print(dataset.shape)


method = TSNE(
    n_components=2, perplexity=30, metric="euclidean"
)  # cannot transform other data instances after training
projected_data = method.fit_transform(preprocessed_dataset1)
plt.scatter(projected_data[:, 0], projected_data[:, 1])
plt.show()
print(dataset.shape)
