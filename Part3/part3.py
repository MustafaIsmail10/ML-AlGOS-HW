import pickle
import numpy as np
import math
import random
import pickle
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
import sklearn.datasets
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.cm as cm

dataset = pickle.load(open("../data/part3_dataset.data", "rb"))


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


def compute_silhouette_score(hac: AgglomerativeClustering, k: int, m):
    """Calculate the avegrage silhouette score of the AgglomerativeClustering give the number of clusters"""
    labels = hac.labels_
    return silhouette_score(dataset, labels, metric=m)


# Preprocessing
max_value = np.max(np.abs(dataset), axis=0)
dataset /= max_value

distance_metric = ["cosine", "euclidean"]
linkage_creterion = ["complete", "single"]

for d in distance_metric:
    for l in linkage_creterion:
        hac = AgglomerativeClustering(
            distance_threshold=0,
            n_clusters=None,
            metric=d,
            linkage=l,
        )
        hac.fit(dataset)
        plot_dendrogram(hac, truncate_mode="level", p=10)
        plt.title(f"distance_metric={d}, linkage_creterion={l}")
        plt.show()

for d in distance_metric:
    for l in linkage_creterion:
        for k in range(2, 6):
            hac = AgglomerativeClustering(
                distance_threshold=None,
                n_clusters=k,
                metric=d,
                linkage=l,
            )
            hac.fit(dataset)

            silhouette_coff = compute_silhouette_score(hac, k, d)
            print(
                f"distance_metric={d}, linkage_creterion={l}, k={k}, silhouette_score={silhouette_coff}"
            )

            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.set_size_inches(18, 7)
            ax1.set_xlim([-0.1, 1])
            ax1.set_ylim([0, len(dataset) + (k + 1) * 10])
            labels = hac.labels_
            sample_silhouette_values = silhouette_samples(dataset, labels, metric=d)

            y_lower = 10
            for i in range(k):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                ith_cluster_silhouette_values = sample_silhouette_values[labels == i]

                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                color = cm.nipy_spectral(float(i) / k)
                ax1.fill_betweenx(
                    np.arange(y_lower, y_upper),
                    0,
                    ith_cluster_silhouette_values,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.7,
                )

                # Label the silhouette plots with their cluster numbers at the middle
                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

            ax1.set_title("The silhouette plot for the various clusters.")
            ax1.set_xlabel("The silhouette coefficient values")
            ax1.set_ylabel("Cluster label")

            # The vertical line for average silhouette score of all the values
            ax1.axvline(x=silhouette_coff, color="red", linestyle="--")

            ax1.set_yticks([])  # Clear the yaxis labels / ticks
            ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

            # 2nd Plot showing the actual clusters formed
            colors = cm.nipy_spectral(labels.astype(float) / k)
            ax2.scatter(
                dataset[:, 0],
                dataset[:, 1],
                marker=".",
                s=30,
                lw=0,
                alpha=0.7,
                c=colors,
                edgecolor="k",
            )

            ax2.set_title("The visualization of the clustered data.")
            ax2.set_xlabel("Feature space for the 1st feature")
            ax2.set_ylabel("Feature space for the 2nd feature")

            plt.suptitle(
                "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
                % k,
                fontsize=14,
                fontweight="bold",
            )
        plt.show()
