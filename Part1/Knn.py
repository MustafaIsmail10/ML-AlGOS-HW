class KNN:
    def __init__(
        self,
        dataset,
        data_label,
        similarity_function,
        similarity_function_parameters=None,
        K=1,
    ):
        """
        :param dataset: dataset on which KNN is executed, 2D numpy array
        :param data_label: class labels for each data sample, 1D numpy array
        :param similarity_function: similarity/distance function, Python function
        :param similarity_function_parameters: auxiliary parameter or parameter array for distance metrics
        :param K: how many neighbors to consider, integer
        """
        self.K = K
        self.dataset = dataset
        self.dataset_label = data_label
        self.similarity_function = similarity_function
        self.similarity_function_parameters = similarity_function_parameters

    def predict(self, instance):
        """
        Predicts the class label of the given instance
        :param instance: test sample, 1D numpy array
        :return: predicted class label, integer
        """
        ## Loop over all the data samples in the dataset and compute the distance and store it in a list with the label as a tuple
        distances = []
        for i in range(len(self.dataset)):
            distances.append(
                (
                    self.similarity_function(
                        instance, self.dataset[i], **self.similarity_function_parameters
                    ),
                    self.dataset_label[i],
                )
            )

        distances.sort(
            key=lambda x: x[0]
        )  ## Sorting according to the distances the distances
        neighbors = distances[: self.K]  ## Selecting the first K neighbors

        labels = [
            neighbor[1] for neighbor in neighbors
        ]  ## Extracting the labels of the neighbors

        return max(set(labels), key=labels.count)  ## Returning the most frequent label
