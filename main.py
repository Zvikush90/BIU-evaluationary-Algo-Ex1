from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
import model
import os
from model import CNN
import ga

def print_gen(i):
    print "=========================================GEN " + str(i)+"========================================="

def run_ga():
    p_count = 20 #100

    p = ga.population(p_count)
    print_gen(0)
    fitness_history = [ga.grade(p), ]
    for i in xrange(100):
        print_gen(i+1)
        p = ga.evolve(p)
        fitness_history.append(ga.grade(p))
    print "=========================================GEN GRADE HISTORY========================================="
    for datum in fitness_history:
        print datum

if __name__ == '__main__':
    # creating folder for files
    newpath = r'./output'
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    run_ga()
    # nb_classes = 10
    # # input image dimensions
    # img_rows, img_cols = 28, 28
    # # # number of convolutional filters to use
    # # nb_filters = 32
    # # # size of pooling area for max pooling
    # # nb_pool = 2
    # # # convolution kernel size
    # # nb_conv = 3
    #
    # # the data, shuffled and split between train and test sets
    # (X_train, y_train), (X_test, y_test) = mnist.load_data()
    #
    # X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    # X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    # X_train = X_train.astype('float32')
    # X_test = X_test.astype('float32')
    # X_train /= 255
    # X_test /= 255
    # print('X_train shape:', X_train.shape)
    # print(X_train.shape[0], 'train samples')
    # print(X_test.shape[0], 'test samples')
    #
    # # convert class vectors to binary class matrices
    # Y_train = to_categorical(y_train, nb_classes)
    # Y_test = to_categorical(y_test, nb_classes)
    #
    # # creating folder for files
    # id = 1
    # newpath = r'./output'
    # if not os.path.exists(newpath):
    #     os.makedirs(newpath)
    #
    # cnn_model = CNN(nb_classes, 10, 2, 0.25, img_rows, img_cols)
    # history = cnn_model.train(id, 0.1, X_train, Y_train)
    # # cnn_model.load_model_from_file(1)
    # cnn_model.test(id, X_test, Y_test)
    # cnn_model.write_model_to_file(id)
    # cnn_model.graph(id, history)
