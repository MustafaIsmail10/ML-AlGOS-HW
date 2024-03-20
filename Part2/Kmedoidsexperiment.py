from Kmedoids import KMedoids
import pickle
import numpy as np
from matplotlib import pyplot as plt


dataset1 = pickle.load(open("../data/part2_dataset_1.data", "rb"))
dataset2 = pickle.load(open("../data/part2_dataset_2.data", "rb"))


dataset1 = np.array(dataset1, dtype=np.float64)
dataset2 = np.array(dataset2, dtype=np.float64)


def run_kmedoids(dataset):
    dataset_results = {}
    for k in range(2, 11):
        loses = []
        for i in range(10):
            min_loss = float("inf")
            for j in range(10):
                kmediods = KMedoids(dataset, K=k)
                (
                    cluster_centers,
                    clusters,
                    loss,
                ) = kmediods.run()
                if loss < min_loss:
                    min_loss = loss

            loses.append(min_loss)

        loses_array = np.array(loses)
        dataset_results[k] = {}
        dataset_results[k]["mean"] = loses_array.mean()
        dataset_results[k]["std"] = loses_array.std()
        dataset_results[k]["confidence_interval"] = 1.96 * (
            dataset_results[k]["std"] / np.sqrt(len(loses))
        )
        print("K: ", k)
        print("Mean: ", dataset_results[k]["mean"])
        print("Std: ", dataset_results[k]["std"])
        print("Confidence Interval: ", dataset_results[k]["confidence_interval"])
        print("=====================================")

    return dataset_results


def plot_results(dataset_results):
    means = []
    confidence_intervals = []
    ks = []
    for k in dataset_results:
        means.append(dataset_results[k]["mean"])
        confidence_intervals.append(dataset_results[k]["confidence_interval"])
        ks.append(k)

    plt.plot(ks, means)
    plt.errorbar(ks, means, yerr=confidence_intervals, fmt="o")
    plt.xlabel("K")
    plt.ylabel("Loss")
    plt.show()


def main():
    print("########### Dataset 1 ###########")
    dataset1_results = run_kmedoids(dataset1)
    print("########### Dataset 2 ###########")
    dataset2_results = run_kmedoids(dataset2)
    print(dataset1_results)
    print(dataset2_results)
    plot_results(dataset1_results)
    plot_results(dataset2_results)


if __name__ == "__main__":
    main()
