""" This file contains code for a fully convolutional
(i.e. contains zero fully connected layers) neural network
for detecting lanes. This version assumes the inputs
to be road images in the shape of 80 x 160 x 3 (RGB) with
the labels as 80 x 160 x 1 (just the G channel with a
re-drawn lane). Note that in order to view a returned image,
the predictions is later stacked with zero'ed R and B layers
and added back to the initial road image.
"""

import numpy as np
import pickle
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# Import necessary items from Keras
from keras.models import Sequential
from keras.layers import Activation, Dropout, UpSampling2D
from keras.layers import Deconvolution2D, Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
import tanuki_net

# Load training images
train_images,labels, _ = pickle.load(open("tanuki_train.p", "rb" ))

# Make into arrays as the neural network wants these
labels = labels[:,1:-1,:-1, np.newaxis]

print("Train_imgs is {}, labels is {}".format(train_images.shape, labels.shape))

# Normalize labels - training images get normalized to start in the network
labels = labels / 255

# Shuffle images along with their labels, then split into training/validation sets
train_images, labels = shuffle(train_images, labels)

# Test size may be 10% or 20%
X_train, X_val, y_train, y_val = train_test_split(train_images, labels, test_size=0.1)

# Batch size, epochs and pool size below are all paramaters to fiddle with for optimization
batch_size = 50
epochs = 1
pool_size = (2, 2)
input_shape = X_train.shape[1:]

# Using a generator to help the model use less data
# I found NOT using any image augmentation here surprisingly yielded slightly better results
# Channel shifts help with shadows but overall detection is worse
datagen = ImageDataGenerator()
datagen.fit(X_train)

# Compiling and training the model
model = tanuki_net.tanuki_net(input_shape, pool_size)
model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size), samples_per_epoch = len(X_train),
                    nb_epoch=epochs, verbose=1, validation_data=(X_val, y_val))

# Save model architecture and weights
model_json = model.to_json()
with open("new_model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights('new_model.h5')