from __future__ import print_function
import numpy as np
import sys

np.random.seed(1337)  # for reproducibility

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, GaussianNoise
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.utils.np_utils import to_categorical

WEIGHTS_FILE = "mnist_cnn_weights.txt"
ID = "204785240.txt"

print(WEIGHTS_FILE)
batch_size = 50
nb_classes = 10
nb_epoch = 500

lr = 0.001
print("Batch Size:" + str(batch_size))
print("Learning Rate: " + str(lr))
# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = [64, 128, 256, 512]
print("Filters: " + str(nb_filters))
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

    # np.save("TrainX", X_train)
    # np.save("TrainY", Y_train)

    return X_train, Y_train


# def train_from_past():
#     X_train = np.load("TrainX.npy")
#     Y_train = np.load("TrainY.npy")
#     return X_train, Y_train


X_train, Y_train = train_from_file()

Y_train = to_categorical(Y_train, nb_classes)


def val_from_file():
    csv = np.genfromtxt('./data/validate1.txt', delimiter=",")
    X_val1 = csv[:, 1:785]
    Y_val1 = csv[:, 0]

    X_val1 = X_val1.reshape(X_val1.shape[0], 1, img_rows, img_cols)

    # np.save("ValX1", X_val1)
    # np.save("ValY1", Y_val1)

    csv = np.genfromtxt('./data/validate2.txt', delimiter=",")
    X_val2 = csv[:, 1:785]
    Y_val2 = csv[:, 0]

    X_val2 = X_val2.reshape(X_val2.shape[0], 1, img_rows, img_cols)

    # np.save("ValX2", X_val2)
    # np.save("ValY2", Y_val2)
    return X_val1, Y_val1, X_val2, Y_val2


# def val_from_past():
#     X_val1 = np.load("ValX1.npy")
#     Y_val1 = np.load("ValY1.npy")
#     X_val2 = np.load("ValX2.npy")
#     Y_val2 = np.load("ValY2.npy")
#     return X_val1, Y_val1, X_val2, Y_val2


X_val1, Y_val1, X_val2, Y_val2 = val_from_file()

Y_val1 = to_categorical(Y_val1, nb_classes)
Y_val2 = to_categorical(Y_val2, nb_classes)

X_train = np.concatenate((X_train, X_val1), axis=0)
Y_train = np.concatenate((Y_train, Y_val1), axis=0)

X_train = np.concatenate((X_train, X_val2), axis=0)
Y_train = np.concatenate((Y_train, Y_val2), axis=0)


def Model(weights_path=None):
    model = Sequential()
    model.add(GaussianNoise(0.01, input_shape=(1, img_rows, img_cols)))
    model.add(Convolution2D(nb_filters[0], nb_conv, nb_conv,
                            border_mode='valid'))
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


def predict_test(model):
    csv = np.genfromtxt('./data/test.txt', delimiter=",")
    X_val = csv[:, 1:785]
    Y_val = csv[:, 0]

    X_val = X_val.reshape(X_val.shape[0], 1, img_rows, img_cols)

    predictions = model.predict_classes(X_val, verbose=True)
    # wrting output to file
    text_file = open("./" + ID, "w")

    for p in predictions:
        text_file.write(str(p) + "\n")

    text_file.close()


def main():
    model = Model()
    if (sys.argv[1] == "test"):
        global nb_epoch
        nb_epoch = 0
        global WEIGHTS_FILE
        WEIGHTS_FILE = sys.argv[2]

    adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam)

    datagen = ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=False)

    datagen.fit(X_train)
    callbacks = [ModelCheckpoint(WEIGHTS_FILE, monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
                 EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')]
    model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                        samples_per_epoch=len(X_train), nb_epoch=nb_epoch, validation_data=(X_val1, Y_val1),
                        show_accuracy=True, callbacks=callbacks)

    model.load_weights(WEIGHTS_FILE)
    predict_test(model)


if __name__ == "__main__":
    main()
