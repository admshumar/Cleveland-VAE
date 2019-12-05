'''
Autoencoder with dense layers and one hidden layer.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.keras.layers import Activation, Dense, Input
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras import optimizers
from data import data_generator
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import numpy as np

# np.random.seed(1337)

def get_power_sequence(n):
    k = len(bin(n)) - 3
    power_sequence = [2**i for i in range(1, k+1)]
    if power_sequence[-1] == n:
        power_sequence=power_sequence[:-1]
    power_sequence.append(n)
    return power_sequence[::-1]

# Load data
def get_data(synthetic=True):
    if synthetic:
        gmm = data_generator.GaussianMixtureData(dimension=16, tightness=0.2)
        data = gmm.data
        dimension = gmm.dimension
    else:
        from numpy import genfromtxt
        data = genfromtxt('../data/cleveland_data.csv', delimiter=',')
        data = data[1:, 2:]
        dimension = len(data[0])
    x_train, x_test = data, data
    # x_train, y_train, x_test, y_test = train_test_split()
    return x_train, x_test, dimension

"""
synthetic=False: Lauge's data
synthetic=True: artificially generated GMM data
"""
x_train, x_test, dimension = get_data(synthetic=False)
power_sequence = get_power_sequence(dimension)

# Network parameters
input_shape = (dimension,)
batch_size = len(x_train)
latent_dim = power_sequence[-1]

# Encoder Structure
inputs = Input(shape=input_shape, name='encoder_input')
z = inputs
# z = Dropout(rate=0.5)(z)
for dim in power_sequence[1:]:
    z = Dense(dim,
              activation='relu'
              )(z)
    z = BatchNormalization()(z)
    #z = Dropout(rate=0.5)(z)

# Instantiate Encoder
encoder = Model(inputs, z, name='encoder')
encoder.summary()

# Decoder Structure
latent_inputs = Input(shape=(latent_dim,), name='decoder_input')
x = latent_inputs
for dim in power_sequence[::-1][1:-1]:
    x = Dense(dim,
              activation='tanh'
              )(x)
    x = BatchNormalization()(x)
    #x = Dropout(rate=0.5)(x)
x = Dense(power_sequence[::-1][-1], activation='tanh')(x)
x = BatchNormalization()(x)
outputs = x

# Instantiate Decoder
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()

# Instantiate Autoencoder Model
autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
autoencoder.summary()
autoencoder.compile(loss='mse',
                    optimizer=optimizers.Adam(lr=1e-3)
                    )

# TensorBoard callbacks
tensorboard_callback = TensorBoard(log_dir='./logs',
                                   histogram_freq=0,
                                   write_graph=True,
                                   write_images=True)

# Train the autoencoder
x_train = (x_train - np.mean(x_train))/np.std(x_train)
early_stopping = EarlyStopping(monitor='loss',
                               min_delta=1e-4,
                               patience=100,
                               restore_best_weights=True)
autoencoder.fit(x_train,
                x_train,
                # validation_data=(x_test, x_test),
                epochs=10000,
                batch_size=batch_size,
                callbacks=[tensorboard_callback, early_stopping])
print("Autoencoder trained.\n")

x_test = (x_test - np.mean(x_test))/np.std(x_test)
x_decoded = autoencoder.predict(x_test)
x_latent = encoder.predict(x_test) # Normalize?


def show(data, kmc=0, gmc=0):
    if latent_dim == 2:
        print("Autoencoded Representation:")
        plt.plot(data[:, 0], data[:, 1], 'o', markersize=1)
        if kmc > 0:
            k_means = KMeans(kmc).fit(data)
            kmc_means = k_means.cluster_centers_
            print("K-Means Clusters:", kmc_means)
            plt.plot(kmc_means[:, 0], kmc_means[:, 1], 'yo', markersize=10)
        if gmc > 0:
            gm = GaussianMixture(gmc).fit(data)
            gmc_means = gm.means_
            print("Gaussian Mixture Clusters:", gmc_means)
            plt.plot(gmc_means[:, 0], gmc_means[:, 1], 'ks', markersize=5)
        plt.axis('equal')
        plt.show()


show(x_latent, 4, 4)
