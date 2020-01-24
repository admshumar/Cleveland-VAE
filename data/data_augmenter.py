import numpy as np
import matplotlib.pyplot as plt


class DataAugmenter:
    """
    A class that augments a data set using a specified probability distribution.
    """
    def __init__(self, data, covariance, number_of_samples=10):
        """
        Initialize a data augmenter with the variance for the Gaussian distributions that sample new data points.
        :param data: The data set to be augmented.
        :param covariance: NumPy array specifying the covariance of the Gaussian distributions that sample new data.
        :param number_of_samples: The number of samples to be drawn from each Gaussian.
        """
        self.data = data
        self.covariance = covariance
        self.number_of_samples = number_of_samples
        self.dimension = len(data[0])

    def sample_near_data_point(self, data_point):
        """
        Generate a list of synthetic data near a given data point.
        :param data_point: The mean of the Gaussian from which new data are sampled.
        :return: A list of synthetic points to be concatenated to the data set.
        """
        list_of_synthetic_points = np.random.multivariate_normal(data_point, self.covariance, self.number_of_samples)
        return list_of_synthetic_points

    def append_list_of_synthetic_data_points(self, list_of_synthetic_points):
        """
        Append a list of synthetically generated data points to the data set.
        :param list_of_synthetic_points: A list of synthetic points to be concatenated to the data set.
        :return: None
        """
        self.data = np.concatenate((self.data, list_of_synthetic_points), axis=0)

    def augment(self):
        """
        Run the augmentation procedure.
        :return: An augmented data set.
        """
        print("Augmenting data.")
        for i in range(len(self.data)):
            list_of_synthetic_points = self.sample_near_data_point(self.data[i])
            self.append_list_of_synthetic_data_points(list_of_synthetic_points)
            del list_of_synthetic_points

        return self.data