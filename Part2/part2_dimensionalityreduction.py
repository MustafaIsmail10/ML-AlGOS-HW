import pickle
import numpy as np
import pickle
import matplotlib.pyplot as plt
import sklearn.datasets
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP


dataset1 = pickle.load(open("../data/part2_dataset_1.data", "rb"))
dataset2 = pickle.load(open("../data/part2_dataset_2.data", "rb"))


color_map = {0: "red", 1: "orange", 2: "green"}

# preprocessing dataset 1
mean1 = np.mean(dataset1, axis=0)
std1 = np.std(dataset1, axis=0)
preprocessed_dataset1 = (dataset1 - mean1) / std1

# preprocessing dataset 2
mean2 = np.mean(dataset2, axis=0)
std2 = np.std(dataset2, axis=0)
preprocessed_dataset2 = (dataset2 - mean2) / std2

################# DATASET 1 #################

method = UMAP(
    n_neighbors=100, n_components=2
)  # can transform other data instances after training

projected_data = method.fit_transform(preprocessed_dataset1)
plt.scatter(projected_data[:, 0], projected_data[:, 1])
plt.show()
print(dataset1.shape)


method = PCA(n_components=2)  # can transform other data instances after training
projected_data = method.fit_transform(preprocessed_dataset1)
plt.scatter(projected_data[:, 0], projected_data[:, 1])
plt.show()
print(dataset1.shape)


method = TSNE(
    n_components=2, perplexity=30, metric="euclidean"
)  # cannot transform other data instances after training
projected_data = method.fit_transform(preprocessed_dataset1)
plt.scatter(projected_data[:, 0], projected_data[:, 1])
plt.show()
print(dataset1.shape)


################# DATASET 2 #################


method = UMAP(
    n_neighbors=100, n_components=2
)  # can transform other data instances after training

projected_data = method.fit_transform(preprocessed_dataset2)
plt.scatter(projected_data[:, 0], projected_data[:, 1])
plt.show()
print(dataset1.shape)


method = PCA(n_components=2)  # can transform other data instances after training
projected_data = method.fit_transform(preprocessed_dataset2)
plt.scatter(projected_data[:, 0], projected_data[:, 1])
plt.show()
print(dataset1.shape)


method = TSNE(
    n_components=2, perplexity=30, metric="euclidean"
)  # cannot transform other data instances after training
projected_data = method.fit_transform(preprocessed_dataset2)
plt.scatter(projected_data[:, 0], projected_data[:, 1])
plt.show()
print(dataset1.shape)
