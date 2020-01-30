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
from data import data_generator, data_augmenter, image_directory_counter
from sklearn.manifold import TSNE
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
import sys

# np.random.seed(1337)

"""
Options for the neural network.
"""
deep = True
is_synthetic = False
number_of_clusters = 3

is_restricted = False
if is_restricted:
    restriction_labels = [1, 2, 3]
    number_of_clusters = len(restriction_labels)

enable_stochastic_gradient_descent = False

has_custom_layers = True
exponent_of_latent_space_dimension = 1

enable_augmentation = False
augmentation_size = 100
covariance_coefficient = 0.2

show_representations = False

def make_image_directory():
    """
    Make a directory for images of latent representations of data produced by the encoder model.
    :return: None
    """
    directory = '/Users/jon/PycharmProjects/Cleveland_VAE/data/images/' + hyperparameter_string
    if not os.path.exists(directory):
        os.makedirs(directory)

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


def get_restricted_data(data):
    for lbl in restriction_labels:
        restricted_data = np.concatenate(tuple(data[data[:, 1] == lbl] for lbl in restriction_labels))
    return np.array(restricted_data)


def generate_data(synthetic=True):
    """
    Retrieve data for input into the model.
    :param synthetic: If True, then Gaussian mixture data are imported from data.data_generator, If false, then
                        a custom data set is brought in.
    :return: A 4-tuple consisting of a train/test split of the data, an integer giving the number of features, and
                        the data.
    """
    if synthetic:
        """
        If the data are chosen to be synthetic, then we grab some synthetic data by instantiating a data generator that
        uses a Gaussian mixture model, we set the variable 'data' equal to the labelled data produced by the generator,
        we get its dimension, and we produce a class-stratified train/test split of the labelled data.
        """
        gmm = data_generator.GaussianMixtureData(dimension=64,
                                                 number_of_clusters=number_of_clusters,
                                                 tightness=0.1,
                                                 cube_side_length=100)
        data = gmm.labelled_data

        data_dim = gmm.dimension
        train, test = train_test_split(data, test_size=0.30, random_state=32, stratify=data[:, 0])

        """
        After we get the split, we concatenate both pieces so that we retain the permutations of the labels produced
        by the split. We then remove the labels from the training data and the test data.
        """
        data = np.concatenate((train, test), axis=0)
        # labels = data[:, 0]
        train = train[:, 1:]
        test = test[:, 1:]

    else:
        """
        If the data are not synthetic, then we're dealing with the ADNI dataset. We want to split these data according
        to class labels, and then create a train/test split.
        """
        from numpy import genfromtxt
        data = genfromtxt('../data/cleveland_data.csv', delimiter=',')

        """
        After the data are gotten from the csv file, we throw away the top row (which corresponds to strings indicating
        the various features) and then create a train-test split that is stratified according to the class label. The
        class labels are found in column 1.
        """
        data = data[1:, ]
        if is_restricted:
            data = get_restricted_data(data)

        train, test = train_test_split(data, test_size=0.30, random_state=16, stratify=data[:, 1])
        data = np.concatenate((train, test), axis=0)
        # labels = data[:, 1]
        train = train[:, 2:]
        test = test[:, 2:]
        data_dim = len(train[0])

    return train, test, data_dim, data


"""
Define two sets of data. One is a set to be left alone until testing. The other is a set to be processed by the
neural network.
"""
x_train, x_test, dimension, original_data = generate_data(synthetic=is_synthetic)
w_train, w_test = x_train, x_test

if has_custom_layers:
    power_sequence = [dimension, 48, 24, 4]
else:
    power_sequence = get_power_sequence(dimension, exponent_of_latent_space_dimension)


"""
Data augmentation.
"""
if enable_augmentation:
    augmenter = data_augmenter.DataAugmenter(x_train,
                                             covariance_coefficient * np.identity(len(x_train[0])),
                                             augmentation_size)
    x_train = augmenter.augment()


def normalize(matrix):
    """
    Define a tensor normalization.
    """
    return (matrix - np.mean(matrix))/np.std(matrix), np.mean(matrix), np.std(matrix)


"""
Hyperparameters for the neural network.
"""
input_shape = (dimension,)
number_of_epochs = 2000
if enable_stochastic_gradient_descent:
    batch_size = 128
else:
    batch_size = len(x_train)
learning_rate = 1e-2
enable_batch_normalization = True
enable_dropout = True
enable_activation = True
encoder_activation = 'sigmoid' # 'relu', 'tanh', 'elu', 'softmax', 'sigmoid'
decoder_activation = 'sigmoid'
dropout_rate = 0.5
l2_constant = 1e-4
patience_limit = number_of_epochs//10
early_stopping_delta = 0.1
latent_dim = power_sequence[-1]


"""
Hyperparameter string for writing files
"""
hyperparameter_list = [number_of_epochs, batch_size, learning_rate, encoder_activation, decoder_activation,
                       enable_batch_normalization, enable_dropout,
                       dropout_rate, l2_constant, patience_limit,
                       early_stopping_delta, latent_dim]
if is_synthetic:
    hyperparameter_list.append("synthetic")

if is_restricted:
    restriction_label_string = ''
    for label in restriction_labels:
        restriction_label_string += str(label)
    hyperparameter_list.append("restricted_{}".format(restriction_label_string))

if enable_augmentation:
    augmentation_string = "_".join(["augmented", str(covariance_coefficient), str(augmentation_size)])
    hyperparameter_list.append(augmentation_string)

if not enable_activation:
    hyperparameter_list.append("PCA")

hyperparameter_string = '_'.join([str(i) for i in hyperparameter_list])

# Count number of directories whose hyperparameters are the same
directory_counter = image_directory_counter.DirectoryCounter(hyperparameter_string)
directory_number = directory_counter.count()

# Append the directory number to the hyperparameter string
hyperparameter_string = '_'.join([hyperparameter_string, 'x{:02d}'.format(directory_number)])

make_image_directory()
log_filename = '../data/images/' + hyperparameter_string + '/experiment.log'
sys.stdout = open(log_filename, "w")

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
        if enable_activation:
            z = Dense(dim,
                      activation=encoder_activation,
                      kernel_regularizer=regularizers.l2(l2_constant)
                      )(z)
        else:
            z = Dense(dim,
                      kernel_regularizer=regularizers.l2(l2_constant)
                      )(z)
        if enable_batch_normalization:
            z = BatchNormalization()(z)
        if enable_dropout:    
            z = Dropout(rate=dropout_rate)(z)
else:
    if enable_activation:
        z = Dense(2,
                  activation=encoder_activation
                  )(z)
    else:
        z = Dense(2)(z)

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
        if enable_activation:
            x = Dense(dim,
                      activation=decoder_activation,
                      kernel_regularizer=regularizers.l2(l2_constant)
                      )(x)
        else:
            x = Dense(dim,
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
    if enable_activation:
        x = Dense(power_sequence[::-1][-1],
              activation=decoder_activation
              )(x)
    else:
        x = Dense(power_sequence[::-1][-1])(x)
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
Data normalizations.
"""
x_train, x_train_mean, x_train_standard_deviation = normalize(x_train)
x_test = (x_test - x_train_mean)/x_train_standard_deviation


"""
Autoencoder training.
"""
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
Produce latent representations of the normalized train data and normalized test data with the encoder.
"""
x_latent_train = encoder.predict(x_train)
x_latent_test = encoder.predict(x_test)


"""
Save the autoencoder, the encoder, and the decoder.
"""
save_model(autoencoder, '/Users/jon/PycharmProjects/Cleveland_VAE/autoencoder')
save_model(decoder, '/Users/jon/PycharmProjects/Cleveland_VAE/decoder')
save_model(encoder, '/Users/jon/PycharmProjects/Cleveland_VAE/encoder')

def get_tsne(data, number_of_components=3):
    """
    Perform t-stochastic neighborhood embedding on a given data set.
    :param data: NumPy array of data to be embedded.
    :return: A t-SNE embedding of the given data.
    """
    perplexity_list = [5, 10, 30, 50]
    for perplexity in perplexity_list:
        embedded_data = TSNE(n_components=number_of_components, perplexity=perplexity).fit_transform(data)
        colors = ['#00B7BA', '#FFB86F', '#5E6572', '#6B0504', '#BA5C12']
        # Tiffany blue, Very light tangelo, Cadet, Rosewood, Alloy orange
        if is_synthetic:
            labels = original_data[:, 0]
        else:
            labels = original_data[:, 1]
        if number_of_components == 2:
            fig_2d = plt.figure(dpi=200)
            plt.scatter(embedded_data[:, 0],
                        embedded_data[:, 1],
                        c=labels[len(w_train):, ],
                        cmap=matplotlib.colors.ListedColormap(colors))
            plt.title('t-SNE with Perplexity {}'.format(perplexity))
            filename = '../data/images/' + hyperparameter_string + '/2d_tsne_{}.png'.format(perplexity)
            fig_2d.savefig(filename)
        else:
            fig_3d = plt.figure(dpi=200)
            ax = fig_3d.add_subplot(projection='3d')
            ax.scatter(embedded_data[:, 0],
                       embedded_data[:, 1],
                       embedded_data[:, 2],
                       c=labels[len(w_train):,],
                       cmap=matplotlib.colors.ListedColormap(colors))

            ax.set_title('t-SNE with Perplexity {}'.format(perplexity))
            filename = '../data/images/' + hyperparameter_string + '/3d_tsne_{}.png'.format(perplexity)
            fig_3d.savefig(filename)

        if show_representations:
            plt.show()

    return embedded_data

def get_kmeans(data, kmc=2):
    """
    Perform k-means clustering on a given data set.
    :param data: NumPy array of data to be clustered.
    :param kmc: Number of clusters centroids to produce.
    :return: A list of cluster centroids and a list of cluster labels for each data point in the data set.
    """
    k_means = KMeans(kmc).fit(data)
    kmc_means = k_means.cluster_centers_
    print("K-Means Clusters:")
    for mean in kmc_means:
        print(mean)
    kmc_latent_labels = k_means.predict(encoder.predict(normalize(w_train)[0]))
    return kmc_means, kmc_latent_labels


def get_gmc(data, gmc=2):
    """
    Perform Gaussian mixture model clustering on a given data set.
    :param data: NumPy array of data to be clustered.
    :param kmc: Number of clusters centroids to produce.
    :return: A list of cluster centroids and a list of cluster labels for each data point in the data set.
    """
    gm = GaussianMixture(gmc).fit(data)
    gmc_means = gm.means_
    print("Gaussian Mixture Clusters:")
    for mean in gmc_means:
        print(mean)
    gmc_latent_labels = gm.predict(encoder.predict(normalize(w_train)[0]))
    return gmc_means, gmc_latent_labels


def aggregate_labels(list_of_labels):
    d = {i: len(list_of_labels[list_of_labels == i])/len(list_of_labels) for i in np.unique(list_of_labels)}
    return d


def show(data, kmc=2, feature_index=6, title=""):
    gmc = kmc
    print("Autoencoded Representation:")

    latent_tnse_2d = get_tsne(data, number_of_components=2)
    latent_tnse_3d = get_tsne(data)
    """
    If kmc > 0, then run K-means clustering on the latent representation of the training data and show
    the cluster centroids that are found.
    """
    if kmc > 0:
        # plt.subplot(2, 3, 1)
        kmc_means, kmc_latent_labels = get_kmeans(data, kmc)
        print("K-Means Labels:", aggregate_labels(kmc_latent_labels), "\n")

    """
    If gmc > 0, then run Gaussian mixture clustering on the latent representation of the training data 
    and show the means that are found.
    """
    if gmc > 0:
        # plt.subplot(2, 3, 1)
        gmc_means, gmc_latent_labels = get_gmc(data, gmc)
        print("Gaussian Mixture Labels:", aggregate_labels(gmc_latent_labels), "\n")

    colors = ['#00B7BA', '#FFB86F', '#5E6572', '#6B0504', '#BA5C12']
    # Tiffany blue, Very light tangelo, Cadet, Rosewood, Alloy orange
    colors = colors[:number_of_clusters]

    if latent_dim == 2:
        print("Autoencoded Representation:")
        fig = plt.figure(dpi=200, figsize=(16, 9))
        if is_synthetic:
            plt.plot(data[:, 0], data[:, 1], markersize=1)
        else:
            if len(data) == len(w_train):
                plt.scatter(data[:, 0], data[:, 1], c=original_data[:len(w_train), 1],
                            cmap=matplotlib.colors.ListedColormap(colors))
                plt.scatter(kmc_means[:, 0], kmc_means[:, 1], marker='P', markersize=10)
                plt.scatter(gmc_means[:, 0], k=gmc_means[:, 1], marker='X', markersize=10)
                # plt.plot([:, 0], kmc_means[:, 1], 'yo', markersize=10)
            if len(data) == len(w_test):
                plt.scatter(data[:, 0], data[:, 1], c=original_data[len(w_train):, 1],
                            cmap=matplotlib.colors.ListedColormap(colors))
                plt.scatter(kmc_means[:, 0], kmc_means[:, 1], marker='X', markersize=10)

    if latent_dim == 4:
        list_of_dimensions = range(0, latent_dim)
        list_of_pairs_of_dimensions = list(itertools.combinations(list_of_dimensions, 2))
        number_of_plots = len(list_of_pairs_of_dimensions)
        number_of_rows = 2
        number_of_columns = 3
        fig, ax = plt.subplots(nrows=number_of_rows, ncols=number_of_columns, sharex=True, dpi=200, figsize=(16, 9))

        for i in range(0, number_of_plots):
            first_dimension = list_of_pairs_of_dimensions[i][0]
            second_dimension = list_of_pairs_of_dimensions[i][1]
            if len(data) == len(w_train):
                ax[i % number_of_rows, i % number_of_columns].scatter(data[:, first_dimension],
                                                                      data[:, second_dimension],
                                                                      c=original_data[:len(w_train), 1],
                                                                      cmap=matplotlib.colors.ListedColormap(colors)
                                                                      )
                ax[i % number_of_rows, i % number_of_columns].scatter(kmc_means[:, first_dimension],
                                                                      kmc_means[:, second_dimension],
                                                                      c='#A6D49F',
                                                                      marker='P')
                ax[i % number_of_rows, i % number_of_columns].scatter(gmc_means[:, first_dimension],
                                                                      gmc_means[:, second_dimension],
                                                                      c='#3AAFB9',
                                                                      marker='d')
                ax[i % number_of_rows, i % number_of_columns].set_title(f'x{str(first_dimension)}, '
                                                                        f'x{str(second_dimension)}')

            if len(data) == len(w_test):
                ax[i % number_of_rows, i % number_of_columns].scatter(data[:, first_dimension],
                                                                      data[:, second_dimension],
                                                                      c=original_data[len(w_train):, 1],
                                                                      cmap=matplotlib.colors.ListedColormap(colors)
                                                                      )
                ax[i % number_of_rows, i % number_of_columns].scatter(kmc_means[:, first_dimension],
                                                                      kmc_means[:, second_dimension],
                                                                      c='#A6D49F',
                                                                      marker='P')
                ax[i % number_of_rows, i % number_of_columns].scatter(gmc_means[:, first_dimension],
                                                                      gmc_means[:, second_dimension],
                                                                      c='#3AAFB9',
                                                                      marker='d')
                ax[i % number_of_rows, i % number_of_columns].set_title(f'x{str(first_dimension)}, '
                                                                        f'x{str(second_dimension)}')

        #if is_synthetic is False:
            """
            Print the mean and standard deviation for each cluster produced from the training set. This reports cluster
            assigments to the training set obtained from the validation set. This is probably bullshit.
            
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

            """
            Print the mean and standard deviation of the set of feature values in the original data set. 
            
            feature = original_data[:, feature_index]
            print("\nTotal mean:", np.mean(feature))
            print("Total Deviation:", np.std(feature))
            """

        plt.axis('equal')

        filename = '../data/images/' + hyperparameter_string + '/2d_'+ str(title) +'.png'
        fig.savefig(filename)
        del filename
        if show_representations:
            plt.show(block=False)

        """
        3D plot of latent variables and a chosen feature.
        """
        if latent_dim == 4:
            fig_3d = plt.figure(dpi=200, figsize=(16, 9))
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
                    ax.scatter(point_array[:, 0], point_array[:, 1], point_array[:, 2],
                               c=original_data[:len(w_train), 1],
                               cmap=matplotlib.colors.ListedColormap(colors))
                ax.scatter(kmc_means[:, 0], kmc_means[:, 1], kmc_means[:, 2],
                           c='#A6D49F', marker='P')

                if len(data) == len(w_test):
                    ax.scatter(point_array[:, 0], point_array[:, 1], point_array[:, 2],
                               c=original_data[len(w_train):, 1],
                               cmap=matplotlib.colors.ListedColormap(colors))
                ax.scatter(gmc_means[:, 0], gmc_means[:, 1], gmc_means[:, 2],
                           c='#3AAFB9', marker='d')

                ax.set_xlabel(f'x{str(first_dimension)}, ')
                ax.set_ylabel(f'x{str(second_dimension)}, ')
                ax.set_zlabel(f'x{str(third_dimension)}')

            filename = '../data/images/' + hyperparameter_string + '/3d_' + str(title) + '.png'
            fig_3d.savefig(filename)

        if show_representations:
            plt.show()
            plt.show(block=False)


show(x_latent_test, kmc=number_of_clusters, feature_index=6, title="x_latent_test")
sys.stdout.close()