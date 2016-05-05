# -*- coding: utf-8 -*-
"""
Created on Wed May 04 16:38:46 2016

@author: User
"""

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.models import model_from_json

# Data related
import matplotlib.pyplot as plt
import os
import numpy as np

batch_size = 128
nb_classes = 10
nb_epoch = 2  # 12
nb_conv, nb_conv = 5, 5  # 48, 5, 5
nb_pool, nb_strides = 2, 2


class CNN:
    def __init__(self, nb_classes, nb_filters, nb_epochs, drop_rate, img_rows, img_cols):
        self.num_classes = nb_classes
        self.nb_filters = nb_filters
        self.nb_epochs = nb_epochs
        self.drop_rate = drop_rate
        self.model = self.get_model(img_rows, img_cols)
        self.score = []

    def get_model(self, img_rows, img_cols):
        model = Sequential()

        # input: nxn images with 1 channel -> (1, n, n) tensors.
        # this applies 48 convolution filters of size 5x5 each.
        model.add(
            Convolution2D(self.nb_filters, nb_conv, nb_conv, border_mode='valid', input_shape=(1, img_rows, img_cols)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool), strides=(nb_strides, nb_strides)))
        model.add(BatchNormalization())

        model.add(Convolution2D(self.nb_filters, nb_conv, nb_conv))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool), strides=(nb_strides, nb_strides)))
        model.add(BatchNormalization())
        model.add(Dropout(self.drop_rate))

        model.add(Flatten())
        # Note: Keras does automatic shape inference.
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dropout(self.drop_rate))

        model.add(Dense(self.num_classes))  # use 2 for binary classification
        model.add(Activation('softmax'))

        return model

    def train(self, lr, X_train, Y_train):
        sgd = SGD(lr, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=sgd)
        history = self.model.fit(X_train, Y_train, batch_size=batch_size,
                                 nb_epoch=self.nb_epochs, verbose=1, show_accuracy=True)

        return history

    def test(self, id, X_test, Y_test):
        self.score = self.model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
        print('Test score:', self.score[0])
        print('Test accuracy:', self.score[1])
        # wrting score to file
        text_file = open("./output/" + str(id) + "_score.txt", "w")
        text_file.write(str(self.score[1]))
        text_file.close()
        return self.score

    def predict(self, input):
        output = self.model.predict(input)
        return output

    def write_model_to_file(self, id):
        print "Saving model to: " + str(os.getcwd())
        json_string = self.model.to_json()
        text_file = open("./output/" + str(id) + "_model.txt", "w")
        text_file.write(json_string)
        text_file.close()

    def load_model_from_file(self, id):
        print "Opening model from: " + str(os.getcwd())
        text_file = open(str(id) + "_model.txt", "r")
        json_string = text_file.read()
        self.model = model_from_json(json_string)
        return self.model

    def graph(self, id, history):
        print str(history.history.keys())
        print str(history.history.values())
        x = np.arange(len(history.epoch))
        vals = history.history.values()
        plt.plot(x, vals[0])
        plt.plot(x, vals[1])
        plt.legend(['acc', 'loss'], loc='upper left')
        #plt.show()
        plt.savefig("./output/"+str(id) + "_fig")


