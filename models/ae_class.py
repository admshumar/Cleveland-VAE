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


class Autoencoder:

    @classmethod
    def normalize(cls, matrix):
        """
        Normalize a NumPy array. (Really unnecessary and should be removed.)
        :param matrix: A NumPy array.
        :return: A standardized NumPy array.
        """
        return (matrix - np.mean(matrix)) / np.std(matrix), np.mean(matrix), np.std(matrix)

    @classmethod
    def aggregate_labels(cls, list_of_labels):
        """
        Construct the class distribution of labels.
        :param list_of_labels: A NumPy array of class labels.
        :return: A dictionary whose keys are classes and whose values are class probabilities.
        """
        d = {i: len(list_of_labels[list_of_labels == i]) / len(list_of_labels) for i in np.unique(list_of_labels)}
        return d

    @classmethod
    def make_image_directory(cls, hyper_parameter_string):
        """
        Make a directory for images of latent representations of data produced by the encoder model.
        :return: None
        """
        directory = '/Users/jon/PycharmProjects/Cleveland_VAE/data/images/' + hyper_parameter_string
        if not os.path.exists(directory):
            os.makedirs(directory)

    @classmethod
    def get_power_sequence(cls, data_dimension, exponent):
        """
        Given an integer, construct a list of integers starting with the given integer, then all positive
        powers of two that are less than the given integer.
        :param data_dimension: An integer that represents the dimension of an input data set.
        :param exponent: The exponent of the dimension of the latent space representation (which is expressed as a
            power of two.)
        :return: A list of integers, which are the dimensions of the feature space representations in the autoencoder.
        """
        k = len(bin(data_dimension)) - 3
        sequence = [2 ** i for i in range(exponent, k + 1)]
        if sequence[-1] == data_dimension:
            sequence = sequence[:-1]
        sequence.append(data_dimension)
        return sequence[::-1]

    @classmethod
    def get_synthetic_data(cls, number_of_clusters):
        """
        Grab some synthetic data by instantiating a Gaussian mixture model data generator, set the variable 'data'
        equal to the labelled data produced by the generator, get its dimension, and produce a class-stratified
        train/test split of the labelled data.
        :param number_of_clusters: An integer indicating the number of Gaussian clusters to be constructed.
        :return: A 4-tuple consisting of a train/test split of the data, an integer giving the number of features, and
            the data.
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
        train = train[:, 1:]
        test = test[:, 1:]

        return train, test, data_dim, data

    @classmethod
    def get_adni_data(cls, is_restricted, restriction_labels):
        """
        Grab the ADNI data set (in .csv form), cast it to a NumPy array, throw away the first row (which consists of
        strings that indicate the features), then do a class-stratified train/test split.
        :param is_restricted: A boolean that indicates whether we ignore any classes during training and validation.
        :param restriction_labels: A list of labels to be kept, provided that we decide to ignore some labels.
        :return: The training data, the test data, the number of features in the ADNI data, and a concatenation of the
            the training data and the test data, which is mainly used to keep track of the permutation of the labels
            produced by the train/test split.
        """
        from numpy import genfromtxt
        data = genfromtxt('../data/cleveland_data.csv', delimiter=',')

        data = data[1:, ]
        if is_restricted:
            for lbl in restriction_labels:
                restricted_data = np.concatenate(tuple(data[data[:, 1] == lbl] for lbl in restriction_labels)) # RED FLAG.
                data = restricted_data

        train, test = train_test_split(data, test_size=0.30, random_state=16, stratify=data[:, 1])
        data = np.concatenate((train, test), axis=0)

        train = train[:, 2:]
        test = test[:, 2:]
        data_dim = len(train[0])

        return train, test, data_dim, data

    @classmethod
    def augment_data(cls, pre_augmented_data, covariance_coefficient, augmentation_size):
        """
        Augments a given data set by placing at each data point a Gaussian whose mean is the data point and then
        sampling from that Gaussian. The covariances of each Gaussian are the same. (It would be interesting to have
        more flexibility for the covariances.)
        :param pre_augmented_data: A NumPy array of data to be augmented.
        :param covariance_coefficient: A float which multiplies the identity covariance matrix.
        :param augmentation_size: A integer indicating the number of points to be sampled.
        :return: A NumPy array of augmented data.
        """
        covariance = covariance_coefficient * np.identity(len(pre_augmented_data[0]))
        augmenter = data_augmenter.DataAugmenter(pre_augmented_data, covariance, augmentation_size)
        return augmenter.augment()

    def __init__(self,
                 deep=True,
                 is_synthetic=False,
                 number_of_clusters=3,
                 is_restricted=False,
                 restriction_labels=[1,2,3],
                 enable_stochastic_gradient_descent=False,
                 has_custom_layers=True,
                 exponent_of_latent_space_dimension=1,
                 enable_augmentation=False,
                 augmentation_size=100,
                 covariance_coefficient=0.2,
                 show_representations=False,
                 number_of_epochs=2000,
                 batch_size=128,
                 learning_rate=1e-3,
                 enable_batch_normalization=True,
                 enable_dropout=True,
                 enable_activation=True,
                 encoder_activation='sigmoid',  # 'relu', 'tanh', 'elu', 'softmax', 'sigmoid'
                 decoder_activation='sigmoid',
                 dropout_rate=0.5,
                 l2_constant=1e-4,
                 early_stopping_delta=0.1
                 ):
        """
        For the ADNI autoencoder, we have options that permit us to size up any symmetric architecture we choose, along
        with whether the data to be autoencoded are from a synthetic data set or a real-world data set. We can decide
        which class labels we want to consider, how to augment the data, and what network hyperparameters we want to
        use. This permits rapid experimentation via repeated instantiations of the class with the desired options and
        hyperparameters. It also provides a method of writing results to directories that quickly indicate the network
        options and hyperparameter values.
        :param deep: A boolean indicating whether the autoencoder has more than one hidden layer.
        :param is_synthetic: A boolean indicating whether the data are generated or come from a real world data set.
        :param number_of_clusters: An integer indicating the number of clusters to be produced by clustering algorithms.
        :param is_restricted: A boolean indicating whether at least one class label is to be ignored.
        :param restriction_labels: A list of integers that indicate the class labels to be retained in the data set.
        :param enable_stochastic_gradient_descent: A boolean indicating whether SGD is performed during training.
        :param has_custom_layers: A boolean indicating the layer structure of the network.
        :param exponent_of_latent_space_dimension: An integer indicating the size of the latent space.
        :param enable_augmentation: A boolean indicating whether data augmentation is to be performed.
        :param augmentation_size: An integer indicating how much data are to be sampled for each existing data point.
        :param covariance_coefficient: A float indicating the scalar multiple of the identity covariance matrix for the
            Gaussians that are used to augment the data.
        :param show_representations: A boolean indicating whether matplotlib.pyplot.show is invoked after an inference
            is performed. By default this is False.
        :param number_of_epochs: An integer indicating the number of training epochs.
        :param batch_size: An integer indicating the batch size.
        :param learning_rate: A float indicating the learning rate.
        :param enable_batch_normalization: A boolean indicating whether batch normalization is performed.
        :param enable_dropout: A boolean indicating whether dropout is performed during training.
        :param enable_activation: A boolean indicating whether activation functions are used during training. In the
            case of an autoencoder, removing network activations will give us an algorithm similar to PCA.
        :param encoder_activation: A boolean indicating the activation function to be used in the encoder layers.
        :param decoder_activation: A boolean indicating the activation function to be used in the decoder layers.
        :param dropout_rate: A float indicating the proportion of neurons to be deactivated.
        :param l2_constant: A float indicating the amount of L2 regularization.
        :param early_stopping_delta: A float indicating the number of epochs before training is halted due to an
            insufficient change in the training loss.
        """
        self.deep = deep
        self.is_synthetic = is_synthetic
        self.is_restricted = is_restricted

        if self.is_restricted:
            self.number_of_clusters = len(self.restriction_labels)
        else:
            self.number_of_clusters = number_of_clusters

        self.enable_stochastic_gradient_descent = enable_stochastic_gradient_descent
        self.has_custom_layers = has_custom_layers
        self.exponent_of_latent_space_dimension = exponent_of_latent_space_dimension
        self.enable_augmentation = enable_augmentation
        self.augmentation_size = augmentation_size
        self.covariance_coefficient = covariance_coefficient
        self.show_representations = show_representations
        self.restriction_labels = restriction_labels

        if self.is_synthetic:
            self.x_train, self.x_test, self.data_dim, self.data = Autoencoder.get_synthetic_data(self.number_of_clusters)
        else:
            self.x_train, self.x_test, self.data_dim, self.data = Autoencoder.get_adni_data(self.is_restricted,
                                                                                            self.restriction_labels)
        self.x_train_length = len(self.x_train)
        self.x_test_length = len(self.x_test)

        if self.is_synthetic:
            self.w_train = self.data[:self.x_train_length, 1:]
            self.w_test = self.data[self.x_train_length:, 1:]
        else:
            self.w_train = self.data[:self.x_train_length, 2:]
            self.w_test = self.data[self.x_train_length:, 2:]
        
        if self.enable_augmentation:
            self.x_train = Autoencoder.augment_data(self.x_train, self.covariance_coefficient, self.augmentation_size)

        """
        Hyperparameters for the neural network.
        """
        self.input_shape = (self.data_dim,)
        self.number_of_epochs = number_of_epochs

        if self.enable_stochastic_gradient_descent:
            self.batch_size = batch_size
        else:
            self.batch_size = len(self.x_train)

        self.learning_rate = learning_rate
        self.enable_batch_normalization = enable_batch_normalization
        self.enable_dropout = enable_dropout
        self.enable_activation = enable_activation
        self.encoder_activation = encoder_activation  # 'relu', 'tanh', 'elu', 'softmax', 'sigmoid'
        self.decoder_activation = decoder_activation
        self.dropout_rate = dropout_rate
        self.l2_constant = l2_constant
        self.patience_limit = self.number_of_epochs // 10
        self.early_stopping_delta = early_stopping_delta

        if self.has_custom_layers:
            self.power_sequence = [self.data_dim, 48, 24, 4]
        else:
            self.power_sequence = self.get_power_sequence(self.data_dim, self.exponent_of_latent_space_dimension)

        self.latent_dim = self.power_sequence[-1]
        
        self.hyper_parameter_list = [self.number_of_epochs, 
                                     self.batch_size, 
                                     self.learning_rate, 
                                     self.encoder_activation, 
                                     self.decoder_activation,
                                     self.enable_batch_normalization, 
                                     self.enable_dropout,
                                     self.dropout_rate, 
                                     self.l2_constant, 
                                     self.patience_limit,
                                     self.early_stopping_delta, 
                                     self.latent_dim]
        if self.is_synthetic:
            self.hyper_parameter_list.append("synthetic")
        
        if self.is_restricted:
            restriction_label_string = ''
            for label in restriction_labels:
                restriction_label_string += str(label)
                self.hyper_parameter_list.append("restricted_{}".format(restriction_label_string))
        
        if self.enable_augmentation:
            augmentation_string = "_".join(["augmented", str(covariance_coefficient), str(augmentation_size)])
            self.hyper_parameter_list.append(augmentation_string)
        
        if not self.enable_activation:
            self.hyper_parameter_list.append("PCA")

        self.hyper_parameter_string = '_'.join([str(i) for i in self.hyper_parameter_list])

        self.directory_counter = image_directory_counter.DirectoryCounter(self.hyper_parameter_string)
        self.directory_number = self.directory_counter.count()
        self.hyper_parameter_string = '_'.join([self.hyper_parameter_string, 'x{:02d}'.format(self.directory_number)])

        """
        Callback to TensorBoard for observing the model structure and network training curves.
        """
        self.tensorboard_callback = TensorBoard(log_dir='./logs',
                                                histogram_freq=0,
                                                write_graph=True,
                                                write_images=True)

        self.early_stopping = EarlyStopping(monitor='val_loss',
                                            min_delta=self.early_stopping_delta,
                                            patience=self.patience_limit,
                                            mode='auto',
                                            restore_best_weights=True)

        self.colors = ['#00B7BA', '#FFB86F', '#5E6572', '#6B0504', '#BA5C12']

        Autoencoder.make_image_directory(self.hyper_parameter_string)
        
    def begin_logging(self):
        log_filename = '../data/images/' + self.hyper_parameter_string + '/experiment.log'
        sys.stdout = open(log_filename, "w")

    def end_logging(self):
        sys.stdout.close()

    def define_encoder(self):
        """
        Structure of an encoder network. Here we have a sequence of Dense maps. The first map takes the input dimension
        to the largest power of two less than the input dimension. The remaining maps halve the dimension. This continues
        until a two-dimensional latent representation is obtained. The Dense maps are composed with batch normalization
        and dropout for network regularization.
        """
        inputs = Input(shape=self.input_shape, name='encoder_input')
        z = inputs
        if self.deep:
            for dimension in self.power_sequence[1:]:
                if self.enable_activation:
                    z = Dense(dimension,
                              activation=self.encoder_activation,
                              kernel_regularizer=regularizers.l2(self.l2_constant)
                              )(z)
                else:
                    z = Dense(dimension,
                              kernel_regularizer=regularizers.l2(self.l2_constant)
                              )(z)
                if self.enable_batch_normalization:
                    z = BatchNormalization()(z)
                if self.enable_dropout:
                    z = Dropout(rate=self.dropout_rate)(z)
        else:
            
            if self.enable_activation:
                z = Dense(2,
                          activation=self.encoder_activation
                          )(z)
            else:
                z = Dense(2)(z)
        
        encoder = Model(inputs, z, name='encoder')
        encoder.summary()
        
        return encoder, inputs

    def define_decoder(self): 
        """
        Structure of a decoder network. Here we have a sequence of Dense maps, doubling the dimension until the largest
        power of two less than the input dimension. The final map returns the input dimension. The Dense maps are composed 
        with batch normalization and dropout for network regularization.
        """
        latent_inputs = Input(shape=(self.latent_dim,), name='decoder_input')
        x = latent_inputs
        if self.deep:
            for dimension in self.power_sequence[::-1][1:]:
                if self.enable_activation:
                    x = Dense(dimension,
                              activation=self.decoder_activation,
                              kernel_regularizer=regularizers.l2(self.l2_constant)
                              )(x)
                else:
                    x = Dense(dimension,
                              kernel_regularizer=regularizers.l2(self.l2_constant)
                              )(x)
                if self.enable_batch_normalization:
                    x = BatchNormalization()(x)
                if self.enable_dropout:
                    x = Dropout(rate=self.dropout_rate)(x)
        else:
            if self.enable_batch_normalization:
                x = BatchNormalization()(x)
            if self.enable_dropout:
                x = Dropout(rate=self.dropout_rate)(x)
            if self.enable_activation:
                x = Dense(self.power_sequence[::-1][-1],
                          activation=self.decoder_activation
                          )(x)
            else:
                x = Dense(self.power_sequence[::-1][-1])(x)
        outputs = x
        decoder = Model(latent_inputs, outputs, name='decoder')
        decoder.summary()
        
        return decoder

    def define_autoencoder(self):
        """
        Structure of an autoencoder, which is the composition of an encoder and a decoder.
        """
        encoder, inputs = self.define_encoder()
        decoder = self.define_decoder()
        auto_encoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
        auto_encoder.summary()
        auto_encoder.compile(loss='mse',
                             optimizer=optimizers.Adam(lr=self.learning_rate)
                             )
        return auto_encoder, encoder, decoder
    
    def fit_autoencoder(self):
        """
        Data normalizations.
        """
        x_train, x_train_mean, x_train_standard_deviation = Autoencoder.normalize(self.x_train)
        x_test = (self.x_test - x_train_mean) / x_train_standard_deviation
        
        auto_encoder, encoder, decoder = self.define_autoencoder()
        auto_encoder.fit(x_train,
                         x_train,
                         validation_data=(x_test, x_test),
                         epochs=self.number_of_epochs,
                         batch_size=self.batch_size,
                         callbacks=[self.tensorboard_callback, self.early_stopping])
        print("Autoencoder trained.\n")
        return auto_encoder, encoder, decoder

    def get_latent_representations(self, encoder):
        return encoder.predict(self.x_train), encoder.predict(self.x_test)

    def save_all_models(self, autoencoder, encoder, decoder):
        """
        Save the autoencoder, the encoder, and the decoder.
        """
        save_model(autoencoder, '/Users/jon/PycharmProjects/Cleveland_VAE/autoencoder')
        save_model(encoder, '/Users/jon/PycharmProjects/Cleveland_VAE/encoder')
        save_model(decoder, '/Users/jon/PycharmProjects/Cleveland_VAE/decoder')

    def get_tsne(self, data, number_of_components=3):
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
            if self.is_synthetic:
                labels = self.data[:, 0]
            else:
                labels = self.data[:, 1]
            if number_of_components == 2:
                fig_2d = plt.figure(dpi=200)
                plt.scatter(embedded_data[:, 0],
                            embedded_data[:, 1],
                            c=labels[self.x_train_length:, ],
                            cmap=matplotlib.colors.ListedColormap(colors))
                plt.title('t-SNE with Perplexity {}'.format(perplexity))
                filename = '../data/images/' + self.hyper_parameter_string + '/2d_tsne_{}.png'.format(perplexity)
                fig_2d.savefig(filename)
            else:
                fig_3d = plt.figure(dpi=200)
                ax = fig_3d.add_subplot(projection='3d')
                ax.scatter(embedded_data[:, 0],
                           embedded_data[:, 1],
                           embedded_data[:, 2],
                           c=labels[self.x_train_length:, ],
                           cmap=matplotlib.colors.ListedColormap(colors))
    
                ax.set_title('t-SNE with Perplexity {}'.format(perplexity))
                filename = '../data/images/' + self.hyper_parameter_string + '/3d_tsne_{}.png'.format(perplexity)
                fig_3d.savefig(filename)
    
            if self.show_representations:
                plt.show()
    
        #return embedded_data

    def get_kmeans(self, data, encoder, kmc=2):
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
        kmc_latent_labels = k_means.predict(encoder.predict(Autoencoder.normalize(self.w_train)[0]))
        return kmc_means, kmc_latent_labels

    def get_gmc(self, data, encoder, gmc=2):
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
        gmc_latent_labels = gm.predict(encoder.predict(self.normalize(self.w_train)[0]))
        return gmc_means, gmc_latent_labels

    def show(self, data, encoder, title='', kmc=4):
        gmc = kmc
        print("Autoencoded Representation:")

        self.get_tsne(data, number_of_components=2)
        self.get_tsne(data)

        """
        Run K-means clustering on the latent representation of the training data and show
        the cluster centroids that are found.
        """
        kmc_means, kmc_latent_labels = self.get_kmeans(data, encoder, kmc)
        print("K-Means Labels:", Autoencoder.aggregate_labels(kmc_latent_labels), "\n")

        """
        Run Gaussian mixture clustering on the latent representation of the training data 
        and show the means that are found.
        """
        gmc_means, gmc_latent_labels = self.get_gmc(data, encoder, gmc)
        print("Gaussian Mixture Labels:", Autoencoder.aggregate_labels(gmc_latent_labels), "\n")

        # Tiffany blue, Very light tangelo, Cadet, Rosewood, Alloy orange
        colors = self.colors[:self.number_of_clusters]

        if self.latent_dim == 2:
            print("Autoencoded Representation:")
            fig = plt.figure(dpi=200, figsize=(16, 9))
            if self.is_synthetic:
                plt.plot(data[:, 0], data[:, 1], markersize=1)
            else:
                if len(data) == self.x_train_length:
                    plt.scatter(data[:, 0], data[:, 1], c=self.data[:self.x_train_length, 1],
                                cmap=matplotlib.colors.ListedColormap(colors))
                    plt.scatter(kmc_means[:, 0], kmc_means[:, 1], marker='P', markersize=10)
                    plt.scatter(gmc_means[:, 0], k=gmc_means[:, 1], marker='X', markersize=10)
                if len(data) == len(self.w_test):
                    plt.scatter(data[:, 0], data[:, 1], c=self.data[self.x_train_length:, 1],
                                cmap=matplotlib.colors.ListedColormap(colors))
                    plt.scatter(kmc_means[:, 0], kmc_means[:, 1], marker='X', markersize=10)

        if self.latent_dim == 4:
            list_of_dimensions = range(0, self.latent_dim)
            list_of_pairs_of_dimensions = list(itertools.combinations(list_of_dimensions, 2))
            number_of_plots = len(list_of_pairs_of_dimensions)
            number_of_rows = 2
            number_of_columns = 3
            fig, ax = plt.subplots(nrows=number_of_rows, ncols=number_of_columns, sharex=True, dpi=200, figsize=(16, 9))

            for i in range(0, number_of_plots):
                first_dimension = list_of_pairs_of_dimensions[i][0]
                second_dimension = list_of_pairs_of_dimensions[i][1]
                if len(data) == self.x_train_length:
                    ax[i % number_of_rows, i % number_of_columns].scatter(data[:, first_dimension],
                                                                          data[:, second_dimension],
                                                                          c=self.data[:self.x_train_length, 1],
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

                if len(data) == len(self.w_test):
                    ax[i % number_of_rows, i % number_of_columns].scatter(data[:, first_dimension],
                                                                          data[:, second_dimension],
                                                                          c=self.data[self.x_train_length:, 1],
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

            plt.axis('equal')

            filename = '../data/images/' + self.hyper_parameter_string + '/2d_' + str(title) + '.png'
            fig.savefig(filename)
            del filename
            if self.show_representations:
                plt.show(block=False)

            """
            3D plot of latent variables and a chosen feature.
            """
            if self.latent_dim == 4:
                fig_3d = plt.figure(dpi=200, figsize=(16, 9))
                list_of_dimensions_3d = range(0, self.latent_dim)
                list_of_triples_of_dimensions_3d = list(itertools.combinations(list_of_dimensions_3d, 3))

                for j in range(len(list_of_dimensions_3d)):
                    first_dimension = list_of_triples_of_dimensions_3d[j][0]
                    second_dimension = list_of_triples_of_dimensions_3d[j][1]
                    third_dimension = list_of_triples_of_dimensions_3d[j][2]
                    ax = fig_3d.add_subplot(2, 2, j + 1, projection='3d')
                    point_array = []
                    for i in range(len(data)):
                        xs = data[i][first_dimension]
                        ys = data[i][second_dimension]
                        zs = data[i][third_dimension]
                        point_array.append([xs, ys, zs])
                    point_array = np.asarray(point_array)

                    if len(data) == self.x_train_length:
                        ax.scatter(point_array[:, 0], point_array[:, 1], point_array[:, 2],
                                   c=self.data[:self.x_train_length, 1],
                                   cmap=matplotlib.colors.ListedColormap(colors))
                    ax.scatter(kmc_means[:, 0], kmc_means[:, 1], kmc_means[:, 2],
                               c='#A6D49F', marker='P')

                    if len(data) == len(self.w_test):
                        ax.scatter(point_array[:, 0], point_array[:, 1], point_array[:, 2],
                                   c=self.data[self.x_train_length:, 1],
                                   cmap=matplotlib.colors.ListedColormap(colors))
                    ax.scatter(gmc_means[:, 0], gmc_means[:, 1], gmc_means[:, 2],
                               c='#3AAFB9', marker='d')

                    ax.set_xlabel(f'x{str(first_dimension)}, ')
                    ax.set_ylabel(f'x{str(second_dimension)}, ')
                    ax.set_zlabel(f'x{str(third_dimension)}')

                filename = '../data/images/' + self.hyper_parameter_string + '/3d_' + str(title) + '.png'
                fig_3d.savefig(filename)

            if self.show_representations:
                plt.show()
                plt.show(block=False)

        plt.close('all')

    def show_latent_representation(self):
        self.begin_logging()
        auto_encoder, encoder, decoder = self.fit_autoencoder()

        x_train_latent, x_test_latent = self.get_latent_representations(encoder)

        self.show(x_test_latent, encoder, title="x_latent_test")

        self.save_all_models(auto_encoder, encoder, decoder)
