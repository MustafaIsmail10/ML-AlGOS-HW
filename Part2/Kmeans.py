import numpy as np
import sys
from random import randint

sys.path.append("..")
from Distance import Distance


class KMeans:
    def __init__(self, dataset, K=2):
        """
        :param dataset: 2D numpy array, the whole dataset to be clustered
        :param K: integer, the number of clusters to form
        """
        self.K = K
        self.dataset = dataset
        # each cluster is represented with an integer index
        # self.clusters stores the data points of each cluster in a dictionary
        self.clusters = {i: [] for i in range(K)}
        # self.cluster_centers stores the cluster mean vectors for each cluster in a dictionary
        self.cluster_centers = {i: None for i in range(K)}
        # you are free to add further variables and functions to the class
        self.dataset_size = len(dataset)

    def calculateLoss(self):
        """Loss function implementation of Equation 1"""
        loss = 0
        for i in range(self.K):
            for p in self.clusters[i]:
                loss += Distance.calculateMinkowskiDistance(
                    self.dataset[p], self.cluster_centers[i], p=2
                )
        return loss

    def run(self):
        """Kmeans algorithm implementation"""
        for i in range(self.K):
            """A heuristic for initializing the cluster centers. Choose two random points and take their mean as the cluster center."""
            p1 = self.dataset[randint(0, len(self.dataset) - 1)]
            p2 = self.dataset[randint(0, len(self.dataset) - 1)]
            self.cluster_centers[i] = (p1 + p2) / 2
        prev_loss = float("inf")
        current_loss = float("inf")
        while True:
            self.clusters = {i: [] for i in range(self.K)}  # reset clusters
            for i in range(self.dataset_size):
                """Assign each point to the closest cluster center"""
                distances_from_cluster_centers = []
                for j in range(self.K):
                    distances_from_cluster_centers.append(
                        (
                            Distance.calculateMinkowskiDistance(
                                self.dataset[i], self.cluster_centers[j], p=2
                            ),
                            j,
                        )
                    )
                distances_from_cluster_centers.sort(
                    key=lambda x: x[0]
                )  # sort by distance from cluster center
                self.clusters[distances_from_cluster_centers[0][1]].append(
                    i
                )  # add the point to the closest cluster

            for i in range(self.K):
                """Calculate the new cluster centers"""
                total = np.zeros(len(self.dataset[0]))
                for p in self.clusters[i]:
                    total += self.dataset[p]

                if len(self.clusters[i]) == 0:
                    self.cluster_centers[i] = total  # type: ignore
                else:
                    self.cluster_centers[i] = total / len(self.clusters[i])  # type: ignore

            current_loss = self.calculateLoss()
            if current_loss == prev_loss:
                break
            else:
                prev_loss = current_loss

        return self.cluster_centers, self.clusters, self.calculateLoss()


def test():
    """
    Testing the functions.
    """
    dataset = np.array([[1, 1], [2, 2], [2, 1], [1, 2], [7, 7], [8, 8], [8, 7], [7, 8]])
    kmeans = KMeans(dataset, 2)
    print(kmeans.run())


if __name__ == "__main__":
    test()
