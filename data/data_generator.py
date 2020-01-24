from scipy.stats import wishart
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt


class GaussianMixtureData:
    """
    A class that generates data from a specified Gaussian mixture model.
    """
    cluster_size = 125

    @classmethod
    def set_cluster_mean(cls, dimension, cube_side_length):
        """
        Choose a cluster centroid using the uniform distribution on an n-dimensional cube.
        :param dimension: Dimension of the ambient space.
        :param cube_side_length: Length of each side of the n-cube.
        :return: A uniformly randomly sampled point on the n-cube.
        """
        a = cube_side_length/2
        cluster_mean = np.random.uniform(-a, a, dimension)
        return cluster_mean

    @classmethod
    def set_cluster_variance(cls, dimension, tightness=5, standard=False):
        """
        Randomly sample a symmetric matrix from a Wishart distribution.
        :param dimension: An integer that specifies the dimension of the data space.
        :param tightness: A float that controls the amount by which a covariance matrix is scaled.
        :param standard: A boolean that determines whether the covariance matrix is the identity matrix or is random.
        :return: A symmetric positive-definite covariance matrix.
        """
        if standard:
            return np.identity(dimension)
        else:
            w = wishart(df=dimension, scale=np.identity(dimension))
            cluster_variance = w.rvs()
            cluster_variance = np.random.uniform(0.1, tightness) * cluster_variance
            return cluster_variance

    @classmethod
    def set_cluster_means(cls, dimension, number_of_clusters, cube_side_length):
        """
        Generate a matrix whose rows are cluster centroids.
        :param dimension: An integer that specifies the dimension of the data space.
        :param number_of_clusters: An integer the specifies the number of cluster centroids.
        :param cube_side_length: Length of each side of the n-cube.
        :return: A matrix of cluser centroids.
        """
        cluster_means = np.empty(shape=(number_of_clusters, dimension))
        for i in range(number_of_clusters):
            cluster_mean = GaussianMixtureData.set_cluster_mean(dimension, cube_side_length)
            cluster_means[i] = cluster_mean
        return cluster_means

    @classmethod
    def set_cluster_variances(cls, dimension, number_of_clusters, tightness=5, standard=False):
        """
        Generate a 3-tensor of covariance matrices indexed by the number of clusters.
        :param dimension: An integer that specifies the dimension of the data space.
        :param number_of_clusters: An integer the specifies the number of cluster centroids.
        :param tightness: A float that controls the amount by which a covariance matrix is scaled.
        :param standard: A boolean that determines whether the covariance matrix is the identity matrix or is random.
        :return: A 3-tensor of covariance matrices indexed by the number of clusters.
        """
        cluster_variances = np.empty(shape=(number_of_clusters, dimension, dimension))
        for i in range(number_of_clusters):
            cluster_variance = GaussianMixtureData.set_cluster_variance(dimension, tightness, standard)
            cluster_variances[i] = cluster_variance
        return cluster_variances

    @classmethod
    def set_cluster(cls, dimension, cluster_mean, cluster_variance, number_of_points=cluster_size):
        """
        Generate a cluster by sampling from a Gaussian.
        :param dimension: An integer that specifies the dimension of the data space.
        :param cluster_mean: The mean of the Gaussian for the cluster.
        :param cluster_variance: The covariance of the Gaussian for the cluser.
        :param number_of_points: The number of points in the cluster.
        :return: A matrix of points belonging to a Gaussian cluster.
        """
        cluster_point_array = np.empty(shape=(number_of_points, dimension))
        for i in range(number_of_points):
            cluster_point = np.random.multivariate_normal(cluster_mean, cluster_variance)
            cluster_point_array[i] = cluster_point
        return cluster_mean, cluster_point_array

    @classmethod
    def set_cluster_size_array(cls, number_of_clusters, random_cluster_size=False):
        """
        Generate an array of numbers of points for a family of clusters.
        :param number_of_clusters: An integer specifying the number of clusters.
        :param random_cluster_size: A boolean indicating whether the number of cluster points varies among clusters.
        :return: A numerical data array of integers specifying the number of cluster points for a family of clusters.
        """
        cluster_size = GaussianMixtureData.cluster_size
        if random_cluster_size:
            cluster_size_array = []
            upper_bound = cluster_size*number_of_clusters
            for i in range(number_of_clusters-1):
                n = np.random.randint(0, upper_bound)
                cluster_size_array.append(n)
                upper_bound = upper_bound - n
            cluster_size_array.append(upper_bound)
        else:
            cluster_size_array = number_of_clusters*[cluster_size]
        return np.asarray(cluster_size_array)

    @classmethod
    def set_clusters(cls, dimension, cluster_means, cluster_variances, random_cluster_size=False):
        """
        Generate a set of clusters and their corresponding sizes.
        :param dimension: An integer that specifies the dimension of the data space.
        :param cluster_means: The set of means of the Gaussian for each cluster.
        :param cluster_variances: The set of covariance matrices of the Gaussian for each cluster.
        :param random_cluster_size: An array of numbers of points for a family of clusters.
        :return: A dictionary whose keys are integer indices and whose values are 2-tuples of
                    cluster means and cluster points.
        """
        clusters = {}
        number_of_clusters = len(cluster_means)
        cluster_sizes = GaussianMixtureData.set_cluster_size_array(number_of_clusters, random_cluster_size)
        for i in range(number_of_clusters):
            cluster_points = GaussianMixtureData.set_cluster(dimension,
                                                             cluster_means[i],
                                                             cluster_variances[i],
                                                             cluster_sizes[i])
            clusters[i] = (cluster_points[0], cluster_points[1])
        return clusters, cluster_sizes

    @classmethod
    def get_data(cls, clusters):
        """
        Concatenate all cluster points into one matrix for input into models.
        :param clusters: A set of dictionaries of cluster data.
        :return: A matrix of data points.
        """
        data = clusters[0][1]
        for i in range(1, len(clusters)):
            data = np.concatenate((data, clusters[i][1]))
        return data

    def __init__(self,
                 dimension=2,
                 number_of_clusters=4,
                 tightness=5,
                 cube_side_length=10,
                 random_cluster_size=False,
                 standard=False):
        """
        Initialize an instance of GaussianMixtureData.
        :param dimension: An integer that specifies the dimension of the data space.
        :param number_of_clusters: An integer specifying the number of clusters.
        :param tightness: A float that controls the amount by which a covariance matrix is scaled.
        :param cube_side_length: Length of each side of the n-cube.
        :param random_cluster_size: An array of numbers of points for a family of clusters.
        :param standard: A boolean that determines whether the covariance matrix is the identity matrix or is random.
        """
        self.standardized = standard
        self.dimension = dimension
        self.number_of_clusters = number_of_clusters
        self.cube_side_length = cube_side_length
        self.cluster_means = GaussianMixtureData.set_cluster_means(self.dimension,
                                                                   self.number_of_clusters,
                                                                   self.cube_side_length)
        self.cluster_variances = GaussianMixtureData.set_cluster_variances(self.dimension,
                                                                           self.number_of_clusters,
                                                                           tightness,
                                                                           self.standardized)
        self.clusters, self.cluster_sizes = GaussianMixtureData.set_clusters(self.dimension,
                                                                             self.cluster_means,
                                                                             self.cluster_variances,
                                                                             random_cluster_size)
        self.total_number_of_points = sum(self.cluster_sizes)
        self.mixture_weights = (1/self.total_number_of_points)*self.cluster_sizes
        self.data = GaussianMixtureData.get_data(self.clusters)

    def k_means(self, number_of_clusters=0):
        """
        Fit a k-means clustering model to the data.
        :param number_of_clusters: Number of clusters for the k-means clustering model.
        :return: The set of means found by the k-means clustering model.
        """
        if number_of_clusters == 0:
            number_of_clusters = self.number_of_clusters
        k_means = KMeans(number_of_clusters).fit(self.data)
        means = k_means.cluster_centers_
        print("K-Means model has been fit.")
        print("Centers:")
        print(means)
        print()
        # self.show(k_means=means)
        return means

    def gmm_means(self, number_of_clusters=0):
        """
        Fit a Gaussian mixture model to the data.
        :param number_of_clusters: Number of clusters for the Gaussian mixture model.
        :return: The set of means found by the Gaussian mixture model.
        """
        if number_of_clusters == 0:
            number_of_clusters = self.number_of_clusters
        gmm_means = GaussianMixture(number_of_clusters).fit(self.data)
        means = gmm_means.means_
        print("Gaussian Mixture model has been fit.")
        print("Centers:")
        print(means)
        print()
        # self.show(gmm_means=means)
        return means

    def show(self, kmc=0, gmm=0):
        """
        Plot all data, including cluster means and means found from Gaussian mixture and k-means clustering models.
        :param kmc: The number of means to find via k-means clustering. If zero, then self.k_means() is not invoked.
        :param gmm: The number of means to fin via a Gaussian mixture. If zero, then self.gmm_means() is not invoked.
        :return: None
        """
        if self.dimension == 2:
            plt.plot(self.data[:, 0], self.data[:, 1], 'o', markersize=1)
            plt.plot(self.cluster_means[:, 0], self.cluster_means[:, 1], 'rs', markersize=6)
            if kmc > 0:
                kmc_means = self.k_means(number_of_clusters=kmc)
                plt.plot(kmc_means[:, 0], kmc_means[:, 1], 'yo', markersize=10)
            if gmm > 0:
                gmm_means = self.gmm_means(number_of_clusters=gmm)
                plt.plot(gmm_means[:, 0], gmm_means[:, 1], 'ks', markersize=5)
            plt.axis('equal')
            plt.show()

    def report(self):
        """
        Print a subset of the attribute dictionary.
        :return: None
        """
        print('Simulated Gaussian Mixture Model Data:',
              '\nDimension:',
              self.dimension,
              '\nNumber of Clusters:',
              self.number_of_clusters,
              '\nCluster Means:',
              self.cluster_means,
              '\nNumber of Points:',
              self.total_number_of_points,
              '\nMixture Weights:',
              self.mixture_weights,
              '\n')

    def show_cluster_map(self):
        return None

"""
def observe(number_of_iterations=1,
            dimension=2,
            number_of_clusters=4,
            tightness=5,
            cube_side_length=10,
            random_cluster_size=False,
            standard=False):
    for i in range(number_of_iterations):
        y = GaussianMixtureData(dimension=dimension,
                                number_of_clusters=number_of_clusters,
                                tightness=tightness,
                                cube_side_length=cube_side_length,
                                random_cluster_size=random_cluster_size,
                                standard=standard)
        print("ITERATION:", i+1)
        y.report()
        y.show(kmc=y.k_means(), gmm=y.gmm_means())
"""

