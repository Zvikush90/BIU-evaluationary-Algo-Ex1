from __future__ import print_function
import numpy as np
from lxml.html import LabelElement

np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.utils.np_utils import to_categorical

# For plotting
import seaborn as sns

batch_size = 10
nb_classes = 10
nb_epoch = 35
# LeNet 0.003 - 96,8 #0.005
lr = 0.005
print("Batch Size:" + str(batch_size))
print("Learning Rate: " + str(lr))
# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = [16,32, 64, 128]
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3


def train_from_file():
    # the data, shuffled and split between train and test sets
    csv = np.genfromtxt('./data/train.txt', delimiter=",")
    X_train = csv[:, 1:785]

    Y_train = csv[:, 0]

    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)

    print('X_train shape:', X_train.shape)
    print('Y_train shape:', Y_train.shape)
    print(X_train.shape[0], 'train samples')

    # X_train, Y_train = augmentation.supplement(X_train, Y_train)
    print("After Augmentation")
    print('X_train shape:', X_train.shape)
    print('Y_train shape:', Y_train.shape)
    print(X_train.shape[0], 'train samples')

    np.save("TrainX", X_train)
    np.save("TrainY", Y_train)

    return X_train, Y_train


def train_from_past():
    X_train = np.load("TrainX.npy")
    Y_train = np.load("TrainY.npy")
    return X_train, Y_train


X_train, Y_train = train_from_past()

Y_train = to_categorical(Y_train, nb_classes)


def val_from_file():
    csv = np.genfromtxt('./data/validate1.txt', delimiter=",")
    X_val = csv[:, 1:785]
    Y_val = csv[:, 0]

    X_val = X_val.reshape(X_val.shape[0], 1, img_rows, img_cols)

    print('X_val shape:', X_val.shape)
    print(X_val.shape[0], 'train samples')
    np.save("ValX", X_val)
    np.save("ValY", Y_val)

    return X_val, Y_val


def val_from_past():
    X_val = np.load("ValX.npy")
    Y_val = np.load("ValY.npy")
    return X_val, Y_val


X_val, Y_val = val_from_past()

Y_val = to_categorical(Y_val, nb_classes)


def Model(weights_path=None):
    model = Sequential()

    model.add(Convolution2D(nb_filters[0], nb_conv, nb_conv,
                            border_mode='valid',
                            input_shape=(1, img_rows, img_cols)))
    model.add(Activation('relu'))

    model.add(Convolution2D(nb_filters[1], nb_conv, nb_conv))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(nb_filters[2], nb_conv, nb_conv))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(nb_filters[3], nb_conv, nb_conv))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model


model = Model()

sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd)

datagen = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=10,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=False)

datagen.fit(X_train)
model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                    samples_per_epoch=len(X_train), nb_epoch=nb_epoch, validation_data=(X_val, Y_val),
                    show_accuracy=True)

model.save_weights(("lenet_l_%f_b_%d_e_%d?weights.txt"%(lr,batch_size,nb_epoch)))
# model.fit(X_train, Y_train, nb_epoch=nb_epoch, batch_size=batch_size, validation_data=(X_val, Y_val), verbose=1,
#         show_accuracy=True, callbacks=[checkpointer])


# model.load_weights("mnist_cnn_weights.txt")




