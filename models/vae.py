### General Imports ###
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

### Autoencoder ###
import tensorflow
import tensorflow.keras

from tensorflow.keras import models, layers
from tensorflow.keras.models import Model, model_from_json

# Get neural network operations
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, UpSampling2D, Input

# Get dataset
from tensorflow.keras.datasets import mnist

# Tensorboard callbacks for Keras
from tensorflow.keras.callbacks import TensorBoard

# Invocation of Tensorboard callback
tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)

### Data Loading ###
(X_train, _), (X_test, _) = mnist.load_data()
shape_x = 28
shape_y = 28

X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.

X_train = X_train.reshape(-1,shape_x,shape_y,1)
X_test = X_test.reshape(-1,shape_x,shape_y,1)

### Model ###
input_img = Input(shape=(shape_x, shape_y, 1))

# Encoding
x = Conv2D(16, (3, 3), padding='same', activation='relu')(input_img)
x = MaxPooling2D(pool_size=(2,2), padding='same')(x)
x = Conv2D(1,(3, 3), padding='same', activation='relu')(x)
encoded = MaxPooling2D(pool_size=(2,2), padding='same')(x)

# Decoding
x = Conv2D(1,(3, 3), padding='same', activation='relu')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16,(3, 3), padding='same', activation='relu')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(1,(3, 3), padding='same')(x)

decoded = Activation('linear')(x)

### Reporting ###
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')
autoencoder.summary()

# Fit the model
autoencoder.fit(X_train, X_train, nb_epoch=15, batch_size=64, validation_split=0.1, callbacks=[tensorboard])

# Save autoencoder weight
json_string = autoencoder.to_json()
autoencoder.save_weights('autoencoder.h5')
open('autoencoder.h5', 'w').write(json_string)

# Build an autoencoder
encoder = Model(inputs = input_img, outputs = encoded)

#Get encoded input
X_train_enc = encoder.predict(X_train)

### Visualize output ###
encoded_imgs = encoder.predict(X_test)
decoded_imgs = autoencoder.predict(X_test)

n = 10
plt.figure(figsize=(20, 4))

for i in range(n):
    # display original
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(X_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Encoded images
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(encoded_imgs[i].reshape(7, 7))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(3, n, i + 1 + 2 * n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()


"""
DENSE VERSION

input_img = Input(shape=(`shape_x * shape_y,))
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)

decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(shape_x * shape_y, activation='sigmoid')(decoded)
"""