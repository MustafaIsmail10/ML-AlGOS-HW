import numpy as np
import math


class Distance:
    @staticmethod
    def calculateCosineDistance(x, y):
        """Implementation of cosine distance as menthionds in the homework pdf

        It computes dot product of x and y and divides it by the product of their norms
        It returns 1 - the resulted value from first computation
        """
        dot_product = np.dot(x, y)
        x_norm = np.linalg.norm(x)  ## computing the norm of x
        y_norm = np.linalg.norm(y)  ## computing the norm of y
        return 1 - (dot_product / (x_norm * y_norm))

    @staticmethod
    def calculateMinkowskiDistance(x, y, p=2):
        """Implementation of Minkowski distance as menthionds in the homework pdf. I used the np.linalg.norm function"""
        return np.linalg.norm(x - y, ord=p)

    @staticmethod
    def calculateMahalanobisDistance(x, y, S_minus_1):
        """Implementation of Mahalanobis distance as menthionds in the homework pdf. I used the np.linalg.norm function"""
        x_minus_y = x - y
        x_minus_y_transpose = np.transpose(x_minus_y)
        return math.sqrt(np.dot(np.dot(x_minus_y_transpose, S_minus_1), x_minus_y))


def test():
    """
    Testing the functions.
    """
    x = np.array([1, 1])
    y = np.array([3, 3])
    print(Distance.calculateCosineDistance(x, y))
    print(Distance.calculateMinkowskiDistance(x, y, p=2))
    print(Distance.calculateMinkowskiDistance(x, y, p=1))
    print(Distance.calculateMahalanobisDistance(x, y, np.identity(2)))


if __name__ == "__main__":
    test()
