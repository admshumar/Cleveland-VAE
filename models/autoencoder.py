'''
Autoencoder with dense layers and one hidden layer.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.keras.layers import Activation, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard
from data import data_generator
from sklearn.model_selection import train_test_split
import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image

#np.random.seed(1337)

# Load data from GMMData
gmm = data_generator.GaussianMixtureData(tightness=0.2, cube_side_length=10)
x_train, y_train, x_test, y_test = train_test_split(gmm.data, gmm.data, test_size=0.20)
dimension = gmm.dimension

# Network parameters
input_shape = (dimension,)
batch_size = gmm.total_number_of_points
latent_dim = int(np.sqrt(dimension))

# Encoder Structure
inputs = Input(shape=input_shape, name='encoder_input')
x = inputs
z = Dense(latent_dim, name='latent_vector')(x)

# Instantiate Encoder
encoder = Model(x, z, name='encoder')
encoder.summary()

# Decoder Structure
latent_inputs = Input(shape=(latent_dim,), name='decoder_input')
x = Dense(dimension)(latent_inputs)
outputs = Activation('sigmoid', name='decoder_output')(x)

# Instantiate Decoder
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()

# Autoencoder = Encoder + Decoder
# Instantiate Autoencoder Model
autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
autoencoder.summary()

autoencoder.compile(loss='mse', optimizer='adam')

# TensorBoard callbacks
tensorboard_callback = TensorBoard(log_dir='./logs',
                                   histogram_freq=0,
                                   write_graph=True,
                                   write_images=True)

# Train the autoencoder
autoencoder.fit(x_train,
                x_train,
                validation_data=(x_test, x_test),
                epochs=500,
                batch_size=batch_size,
                callbacks=[tensorboard_callback])
print("Autoencoder trained.\n")

x_decoded = autoencoder.predict(x_test)

gmm.report()
gmm.show(gmm.number_of_clusters, gmm.number_of_clusters)

"""
# Display the 1st 8 corrupted and denoised images
rows, cols = 10, 30
num = rows * cols
imgs = np.concatenate([x_test[:num], x_test_noisy[:num], x_decoded[:num]])
imgs = imgs.reshape((rows * 3, cols, dimension, dimension))
imgs = np.vstack(np.split(imgs, rows, axis=1))
imgs = imgs.reshape((rows * 3, -1, dimension, dimension))
imgs = np.vstack([np.hstack(i) for i in imgs])
imgs = (imgs * 255).astype(np.uint8)
plt.figure()
plt.axis('off')
plt.title('Original images: top rows, '
          'Corrupted Input: middle rows, '
          'Denoised Input:  third rows')
plt.imshow(imgs, interpolation='none', cmap='gray')
Image.fromarray(imgs).save('corrupted_and_denoised.png')
plt.show()
"""