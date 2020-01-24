'''
Autoencoder with dense layers and one hidden layer.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.keras.layers import Activation, Dense, Input
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras.models import Model, save_model
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras import optimizers
from data import data_generator, data_augmenter
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy.special import comb
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import itertools
import os

# np.random.seed(1337)

"""
Options for the neural network.
"""
deep = True
is_synthetic = False
enable_augmentation = True
augmentation_size = 250
show_representations = False
covariance_coefficient = 0.2
exponent_of_latent_space_dimension = 1

def get_power_sequence(n, exponent_of_latent_space_dimension=1):
    """
    Given an integer, construct a list of integers starting with the given integer, then all positive
    powers of two that are less than the given integer.
    :param n:
    :return: A list of integers.
    """
    k = len(bin(n)) - 3
    sequence = [2**i for i in range(exponent_of_latent_space_dimension, k+1)]
    if sequence[-1] == n:
        sequence = sequence[:-1]
    sequence.append(n)
    return sequence[::-1]


def generate_data(synthetic=True):
    """
    Retrieve data for input into the model.
    :param synthetic: If True, then Gaussian mixture data are imported from data.data_generator, If false, then
                        a custom data set is brought in.
    :return: A 4-tuple consisting of a train/test split of the data, an integer giving the number of features, and
                        the data.
    """
    if synthetic:
        gmm = data_generator.GaussianMixtureData(dimension=8, tightness=0.2, cube_side_length=20)
        data = gmm.data
        data_dim = gmm.dimension
        train, test = train_test_split(data, test_size=0.30, random_state=32)
    else:
        """
        If the data are not synthetic, then we're dealing with the ADNI dataset. We want to split these data according
        to class labels, create train test
        """
        from numpy import genfromtxt
        data = genfromtxt('../data/cleveland_data.csv', delimiter=',')
        """
        After the data are gotten from the csv file, we throw away the top row (which corresponds to strings indicating
        the various features) and then create a train-test split that is stratified according to the class label.
        """
        data = data[1:, ]
        train, test = train_test_split(data, test_size=0.30, random_state=16, stratify=data[:, 1])
        data = np.concatenate((train, test), axis=0)
        train = train[:, 2:]
        test = test[:, 2:]
        data_dim = len(train[0])
    return train, test, data_dim, data


"""
Define two sets of data. One is a set to be left alone until testing. The other is a set to be processed by the
neural network.
"""
w_train, w_test, w_dim, original_data = generate_data(synthetic=is_synthetic)
x_train, x_test, dimension, original_data = generate_data(synthetic=is_synthetic)
power_sequence = get_power_sequence(dimension, exponent_of_latent_space_dimension)
if enable_augmentation:
    augmenter = data_augmenter.DataAugmenter(x_train, covariance_coefficient * np.identity(len(x_train[0])), augmentation_size)
    x_train = augmenter.augment()

"""
Define a tensor normalization.
"""
def normalize(matrix):
    return (matrix - np.mean(matrix))/np.std(matrix)


"""
Hyperparameters for the neural network.
"""
input_shape = (dimension,)
number_of_epochs = 2500
batch_size = 1000
learning_rate = 1e-3
enable_batch_normalization = False
enable_dropout = False
dropout_rate = 0.25
l2_constant = 0 #1e-4
patience_limit = 100
early_stopping_delta = 0.01
latent_dim = power_sequence[-1]

"""
Hyperparameter string for writing files
"""
hyperparameter_list = [number_of_epochs, batch_size, learning_rate,
                       enable_batch_normalization, enable_dropout,
                       dropout_rate, l2_constant, patience_limit,
                       early_stopping_delta, latent_dim]
if is_synthetic:
    hyperparameter_list.append("synthetic")
if enable_augmentation:
    augmentation_string = "_".join(["augmented", str(covariance_coefficient), str(augmentation_size)])
    hyperparameter_list.append(augmentation_string)

hyperparameter_string = '_'.join([str(i) for i in hyperparameter_list])

"""
Structure of an encoder network. Here we have a sequence of Dense maps. The first map takes the input dimension
to the largest power of two less than the input dimension. The remaining maps halve the dimension. This continues
until a two-dimensional latent representation is obtained. The Dense maps are composed with batch normalization
and dropout for network regularization.
"""
inputs = Input(shape=input_shape, name='encoder_input')
z = inputs
if deep:
    for dim in power_sequence[1:]:
        z = Dense(dim,
                  activation='relu',
                  kernel_regularizer=regularizers.l2(l2_constant)
                  )(z)
        if enable_batch_normalization:
            z = BatchNormalization()(z)
        if enable_dropout:    
            z = Dropout(rate=dropout_rate)(z)
else:
    z = Dense(2,
              activation='tanh'
              )(z)
encoder = Model(inputs, z, name='encoder')
encoder.summary()

"""
Structure of a decoder network. Here we have a sequence of Dense maps, doubling the dimension until the largest
power of two less than the input dimension. The final map returns the input dimension. The Dense maps are composed 
with batch normalization and dropout for network regularization.
"""
latent_inputs = Input(shape=(latent_dim,), name='decoder_input')
x = latent_inputs
if deep:
    for dim in power_sequence[::-1][1:]:
        x = Dense(dim,
                  activation='tanh',
                  kernel_regularizer=regularizers.l2(l2_constant)
                  )(x)
        if enable_batch_normalization:
            x = BatchNormalization()(x)
        if enable_dropout:
            x = Dropout(rate=dropout_rate)(x)
else:
    if enable_batch_normalization:
        x = BatchNormalization()(x)
    if enable_dropout:
        x = Dropout(rate=dropout_rate)(x)
    x = Dense(power_sequence[::-1][-1],
              activation='tanh'
              )(x)
outputs = x
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()


"""
Structure of an autoencoder, which is the composition of an encoder and a decoder.
"""
autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
autoencoder.summary()
autoencoder.compile(loss='mse',
                    optimizer=optimizers.Adam(lr=learning_rate)
                    )


"""
Callback to TensorBoard for observing the model structure and network training curves.
"""
tensorboard_callback = TensorBoard(log_dir='./logs',
                                   histogram_freq=0,
                                   write_graph=True,
                                   write_images=True)

"""
Data augmentation.
"""
def augment(data):
    """
    Takes a dataset and generates synthetic Gaussian data from it. Each data point in the original data set is the mean,
    and the variance is a parameter to be set by the user.
    :param data: The input data set to be augmented.
    :return: An augmented data set.
    """


    return data


"""
Training of the autoencoder.
"""
x_train = normalize(x_train)
early_stopping = EarlyStopping(monitor='val_loss',
                               min_delta=early_stopping_delta,
                               patience=patience_limit,
                               mode='auto',
                               restore_best_weights=True)
autoencoder.fit(x_train,
                x_train,
                validation_data=(x_test, x_test),
                epochs=number_of_epochs,
                batch_size=batch_size,
                callbacks=[tensorboard_callback, early_stopping])
print("Autoencoder trained.\n")



"""
Produce latent representations of the train data and test data with the encoder.
"""
x_test = normalize(x_test)
x_latent_train = encoder.predict(x_train)
x_latent_test = encoder.predict(x_test)


"""
Save the autoencoder and the encoder
"""
save_model(autoencoder, '/Users/jon/PycharmProjects/Cleveland_VAE/autoencoder')
save_model(decoder, '/Users/jon/PycharmProjects/Cleveland_VAE/decoder')
save_model(encoder, '/Users/jon/PycharmProjects/Cleveland_VAE/encoder')


def get_kmeans(data, kmc=2):
    k_means = KMeans(kmc).fit(data)
    kmc_means = k_means.cluster_centers_
    print("K-Means Clusters:", kmc_means)
    plt.plot(kmc_means[:, 0], kmc_means[:, 1], 'yo', markersize=10)
    kmc_latent_labels = k_means.predict(encoder.predict(normalize(w_train)))
    return kmc_means, kmc_latent_labels


def get_gmc(data, gmc=2):
    gm = GaussianMixture(gmc).fit(data)
    gmc_means = gm.means_
    print("Gaussian Mixture Clusters:", gmc_means)
    plt.plot(gmc_means[:, 0], gmc_means[:, 1], 'ks', markersize=5)
    gmc_latent_labels = gm.predict(encoder.predict(normalize(w_train)))
    return gmc_means, gmc_latent_labels


def make_image_directory():
    directory = '/Users/jon/PycharmProjects/Cleveland_VAE/data/images/' + hyperparameter_string
    if not os.path.exists(directory):
        os.makedirs(directory)


def show(data, kmc=2, feature_index=6, title=""):
    gmc = kmc
    make_image_directory()
    colors = ['yellow', 'skyblue', 'deepskyblue', 'steelblue']

    if latent_dim == 2:
        print("Autoencoded Representation:")
        fig = plt.figure()
        if is_synthetic:
            plt.plot(data[:, 0], data[:, 1], markersize=1)
        else:
            if len(data) == len(w_train):
                plt.scatter(data[:, 0], data[:, 1], c=original_data[:len(w_train), 1],
                            cmap=matplotlib.colors.ListedColormap(colors))
            if len(data) == len(w_test):
                plt.scatter(data[:, 0], data[:, 1], c=original_data[len(w_train):, 1],
                            cmap=matplotlib.colors.ListedColormap(colors))

    if latent_dim == 4:
        print("Autoencoded Representation:")
        """
        If kmc > 0, then run K-means clustering on the latent representation of the training data and show
        the cluster centroids that are found.
        """
        if kmc > 0:
            # plt.subplot(2, 3, 1)
            kmc_means, kmc_latent_labels = get_kmeans(data, kmc)
            print("K-Means Labels:", kmc_latent_labels)

        """
        If gmc > 0, then run Gaussian mixture clustering on the latent representation of the training data 
        and show the means that are found.
        """
        if gmc > 0:
            # plt.subplot(2, 3, 1)
            gmc_means, gmc_latent_labels = get_gmc(data, gmc)
            print("Gaussian Mixture Labels:", gmc_latent_labels)

        list_of_dimensions = range(0, latent_dim)
        list_of_pairs_of_dimensions = list(itertools.combinations(list_of_dimensions, 2))
        number_of_plots = len(list_of_pairs_of_dimensions)
        number_of_rows = 2
        number_of_columns = 3
        fig, ax = plt.subplots(nrows=number_of_rows, ncols=number_of_columns, sharex=True) #figsize=[10, 5], dpi=300

        for i in range(0, number_of_plots):
            first_dimension = list_of_pairs_of_dimensions[i][0]
            second_dimension = list_of_pairs_of_dimensions[i][1]
            if is_synthetic:
                ax[i % number_of_rows, i % number_of_columns].scatter(data[:, first_dimension],
                                                                      data[:, second_dimension]
                                                                      )
                ax[i % number_of_rows, i % number_of_columns].set_title(f'x{str(first_dimension)}, '
                                                                        f'x{str(second_dimension)}')
            else:
                if len(data) == len(w_train):
                    ax[i % number_of_rows, i % number_of_columns].scatter(data[:, first_dimension],
                                                                          data[:, second_dimension],
                                                                          c=original_data[:len(w_train), 1],
                                                                          cmap=matplotlib.colors.ListedColormap(colors)
                                                                          )
                    ax[i % number_of_rows, i % number_of_columns].set_title(f'x{str(first_dimension)}, '
                                                                            f'x{str(second_dimension)}')

                if len(data) == len(w_test):
                    ax[i % number_of_rows, i % number_of_columns].scatter(data[:, first_dimension],
                                                                          data[:, second_dimension],
                                                                          c=original_data[len(w_train):, 1],
                                                                          cmap=matplotlib.colors.ListedColormap(colors)
                                                                          )
                    ax[i % number_of_rows, i % number_of_columns].set_title(f'x{str(first_dimension)}, '
                                                                            f'x{str(second_dimension)}')

        if is_synthetic is False:
            """
            Print the mean and standard deviation for each cluster produced from the training set.
            """
            unique_labels = np.unique(kmc_latent_labels)
            for label in unique_labels:
                print("\nLabel:", label)
                indices = np.where(kmc_latent_labels == label)
                indices = indices[0]
                cluster = [w_train[i] for i in indices]
                mn = np.mean(cluster, 0)  # compute the mean for each feature
                dv = np.std(cluster, 0)  # compute the standard deviation for each feature
                print("Cluster Size:", len(cluster))
                print("Mean:", mn[feature_index])
                print("Deviation:", dv[feature_index])
                del cluster
                del mn
                del dv

            """
            Print the mean and standard deviation of the set of feature values in the original data set. 
            """
            feature = original_data[:, feature_index]
            print("\nTotal mean:", np.mean(feature))
            print("Total Deviation:", np.std(feature))

        plt.axis('equal')

        filename = '../data/images/' + hyperparameter_string + '/2d_'+ str(title) +'.png'
        fig.savefig(filename)
        del filename
        plt.show(block=False)

        """
        3D plot of latent variables and a chosen feature.
        """
        if latent_dim == 4:
            fig_3d = plt.figure()
            list_of_dimensions_3d = range(0, latent_dim)
            list_of_triples_of_dimensions_3d = list(itertools.combinations(list_of_dimensions_3d, 3))

            for j in range(len(list_of_dimensions_3d)):
                first_dimension = list_of_triples_of_dimensions_3d[j][0]
                second_dimension = list_of_triples_of_dimensions_3d[j][1]
                third_dimension = list_of_triples_of_dimensions_3d[j][2]
                ax = fig_3d.add_subplot(2, 2, j+1, projection='3d')
                point_array = []
                for i in range(len(data)):
                    xs = data[i][first_dimension]
                    ys = data[i][second_dimension]
                    zs = data[i][third_dimension]
                    point_array.append([xs, ys, zs])
                point_array = np.asarray(point_array)
                if len(data) == len(w_train):
                    ax.scatter(point_array[:,0], point_array[:,1], point_array[:,2],
                               c=original_data[:len(w_train), 1],
                               cmap=matplotlib.colors.ListedColormap(colors))
                if len(data) == len(w_test):
                    ax.scatter(point_array[:,0], point_array[:,1], point_array[:,2],
                               c=original_data[len(w_train):, 1],
                               cmap=matplotlib.colors.ListedColormap(colors))
                ax.set_xlabel(f'x{str(first_dimension)}, ')
                ax.set_ylabel(f'x{str(second_dimension)}, ')
                ax.set_zlabel(f'x{str(third_dimension)}')

            filename = '../data/images/' + hyperparameter_string + '/3d_' + str(title) + '.png'
            fig_3d.savefig(filename)

        plt.show()
        plt.show(block=False)



show(x_latent_test, kmc=4, feature_index=6, title="x_latent_test")
show(x_latent_train, kmc=4, feature_index=6, title="x_latent_train")
# show(x_latent_test, kmc=4, feature_index=14)
# show(x_latent_train, kmc=4, feature_index=14)

