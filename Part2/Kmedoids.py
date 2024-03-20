import sys
import numpy as np
from random import randint

sys.path.append("..")
from Distance import Distance


class KMedoids:
    def __init__(self, dataset, K=2, distance_metric="cosine"):
        """
        :param dataset: 2D numpy array, the whole dataset to be clustered
        :param K: integer, the number of clusters to form
        """
        self.K = K
        self.dataset = dataset
        self.distance_metric = distance_metric
        # each cluster is represented with an integer index
        # self.clusters stores the data points of each cluster in a dictionary
        # In this dictionary, you can keep either the data instance themselves or their corresponding indices in the dataset (self.dataset).
        self.clusters = {i: [] for i in range(K)}
        # self.cluster_medoids stores the cluster medoid for each cluster in a dictionary
        # # In this dictionary, you can keep either the data instance themselves or their corresponding indices in the dataset (self.dataset).
        self.cluster_medoids = {i: None for i in range(K)}
        # you are free to add further variables and functions to the class
        self.dataset_size = len(dataset)
        self.distance_functions = {
            "cosine": Distance.calculateCosineDistance,
            "eculidean": Distance.calculateMinkowskiDistance,
        }
        self.distance_function = self.distance_functions[self.distance_metric]

    def calculateLoss(self):
        """Loss function implementation of Equation 2"""
        loss = 0
        for i in range(self.K):
            for p in self.clusters[i]:
                loss += self.distance_function(
                    self.dataset[p], self.dataset[self.cluster_medoids[i]]
                )
        return loss

    def run(self):
        """Kmedoids algorithm implementation"""
        for k in range(self.K):
            """Initializing the cluster medoids. Choose a random point as the cluster medoid."""
            self.cluster_medoids[k] = randint(0, self.dataset_size - 1)  # type: ignore

        ## Calculate the distance matrix
        distance_matrix = np.zeros((self.dataset_size, self.dataset_size))
        for i in range(self.dataset_size):
            for j in range(i + 1, self.dataset_size):
                distance_matrix[i, j] = self.distance_function(
                    self.dataset[i], self.dataset[j]
                )
                distance_matrix[j, i] = distance_matrix[i, j]

        prev_loss = 0
        current_loss = 0
        while True:
            self.clusters = {i: [] for i in range(self.K)}
            for i in range(self.dataset_size):
                """Assign each point to the closest cluster medoid"""
                distances_from_cluster_medoids = []
                for k in range(self.K):
                    distances_from_cluster_medoids.append(
                        (
                            distance_matrix[i, self.cluster_medoids[k]],
                            k,
                        )
                    )
                distances_from_cluster_medoids.sort(key=lambda x: x[0])
                self.clusters[distances_from_cluster_medoids[0][1]].append(i)

            for i in range(self.K):
                """Update the cluster medoids"""
                min_loss = float("inf")
                for p in self.clusters[i]:
                    loss = 0
                    for q in self.clusters[i]:
                        loss += distance_matrix[p][q]
                    if loss < min_loss:
                        min_loss = loss
                        self.cluster_medoids[i] = p

            current_loss = self.calculateLoss()
            if current_loss == prev_loss:
                break
            else:
                prev_loss = current_loss

        return self.cluster_medoids, self.clusters, self.calculateLoss()


def test():
    """
    Testing the functions.
    """
    dataset = np.array([[1, 1], [2, 2], [2, 1], [1, 2], [7, 7], [8, 8], [8, 7], [7, 8]])
    kmedoids = KMedoids(dataset, K=2)
    print(kmedoids.run())


if __name__ == "__main__":
    test()
