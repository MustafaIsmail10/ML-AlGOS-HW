import pickle
import importlib
from Knn import KNN
import sys
from sklearn.model_selection import RepeatedStratifiedKFold
import numpy as np

sys.path.append("..")
from Distance import Distance


dataset, labels = pickle.load(open("../data/part1_dataset.data", "rb"))
kfold = RepeatedStratifiedKFold(n_splits=10, n_repeats=5)


hyperparameters = {
    "k": [3, 5, 7, 9],
    "distance": [
        Distance.calculateCosineDistance,
        Distance.calculateMinkowskiDistance,
    ],
    "params": [{}, {"p": 1}],
}

for k in hyperparameters["k"]:
    for distance_index in range(len(hyperparameters["distance"])):
        distance = hyperparameters["distance"][distance_index]
        parameters = hyperparameters["params"][distance_index]
        print("K: ", k)
        print("Distance: ", distance.__name__)
        print("Parameters: ", parameters)
        print("=====================================")
        accuracy_list = []
        i = 0
        for train_indices, test_indices in kfold.split(dataset, labels):
            current_train = dataset[train_indices]
            current_train_label = labels[train_indices]

            knn = KNN(
                dataset=current_train,
                K=k,
                data_label=current_train_label,
                similarity_function=distance,
                similarity_function_parameters=parameters,
            )

            results = []
            for test_instance_index in test_indices:
                prediction = knn.predict(dataset[test_instance_index])
                if prediction == labels[test_instance_index]:
                    results.append(1)
                else:
                    results.append(0)

            accuracy_list.append(sum(results) / len(results))

        accuracy_array = np.array(accuracy_list)
        accuracy_average = np.average(accuracy_array)
        accuracy_std = np.std(accuracy_array)
        accuracy_confidence_interval = 1.96 * (
            accuracy_std / np.sqrt(len(accuracy_list))
        )
        print("Accuracy: %.2f" % accuracy_average)
        print("Standard Deviation: %.2f" % accuracy_std)
        print("Confidence Interval: %.2f" % accuracy_confidence_interval)
        print("=====================================")
        print()
