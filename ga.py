"""
# Example usage
from genetic import *
target = 371
p_count = 100
i_length = 6
i_min = 0
i_max = 100
p = population(p_count, i_length, i_min, i_max)
fitness_history = [grade(p, target),]
for i in xrange(100):
    p = evolve(p, target)
    fitness_history.append(grade(p, target))

for datum in fitness_history:
   print datum
"""
import random
from random import randint
from operator import add

from scipy.stats._continuous_distns import genexpon_gen

from model import CNN
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical

nb_classes = 10
# input image dimensions
img_rows, img_cols = 28, 28
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
Y_train = to_categorical(y_train, nb_classes)
Y_test = to_categorical(y_test, nb_classes)

gen_number = 0

def individual():
    'Create a member of the population.'
    # learning rate, nb filters, nb epochs, dropout rate
    return [round(random.uniform(0.001, 0.6), 3), randint(1, 5), randint(1, 3),
            round(random.uniform(0.001, 0.5), 3)]  # randint(1,50), randint(1,12)


def population(count):
    """
    Create a number of individuals (i.e. a population).

    count: the number of individuals in the population
    """
    return [(individual(),i) for i in xrange(count)]


def fitness(individual):
    """
    Determine the fitness of an individual. Higher is better.

    individual: the individual to evaluate
    target: the target number individuals are aiming for
    """
    arr, id = individual
    id = str(gen_number)+"_"+str(id)
    print "Current Individual: " + id
    print "Learning rate - " + str(arr[0])+", Nb filters - " + str(arr[1]) + ", Nb epochs - "\
          + str(arr[2]) + ", Dropout rate - " + str(arr[3])
    cnn_model = CNN(nb_classes, arr[1], arr[2], arr[3], img_rows, img_cols)
    history = cnn_model.train(arr[0], X_train, Y_train)
    # cnn_model.load_model_from_file(1)

    score = cnn_model.test(id, X_test, Y_test)
    cnn_model.write_model_to_file(id)
    #cnn_model.graph(id, history)

    # text_file = open(str(id) + "_score.txt", "r")
    # score = text_file.read()
    # text_file.close()
    return score[1]


def grade(pop):
    'Find average fitness for a population.'
    fit_list = [fitness(x) for x in pop]
    print "FITNESS LIST OF GEN: " + str(sorted(fit_list))
    summed = sum(fit_list)
    result = summed / (len(pop) * 1.0)
    print "GEN GRADE = "+str(result)


def evolve(pop, retain=0.2, random_select=0.05, mutate=0.01):
    global gen_number
    gen_number = gen_number+1
    graded = [(fitness(x), x) for x in pop]
    graded = [x[1] for x in sorted(graded)]
    retain_length = int(len(graded) * retain)
    parents = graded[:retain_length]
    # randomly add other individuals to
    # promote genetic diversity
    for individual in graded[retain_length:]:
        from random import random
        if random_select > random():
            parents.append(individual)

    # crossover parents to create children
    parents_length = len(parents)
    desired_length = len(pop) - parents_length
    children = []
    while len(children) < desired_length:
        male = randint(0, parents_length - 1)
        female = randint(0, parents_length - 1)
        if male != female:
            male = parents[male]
            female = parents[female]
            half = len(male) / 2
            child = male[:half] + female[half:]
            children.append(child)
    parents.extend(children)

    # mutate some individuals
    for individual in parents:
        if mutate > random():
            pos_to_mutate = randint(0, len(individual) - 1)
            if (pos_to_mutate == 0):
                round(random.uniform(0.001, 0.6), 3)
            elif (pos_to_mutate == 1):
                randint(1, 5)
            elif (pos_to_mutate == 2):
                randint(1, 3)
            else:
                round(random.uniform(0.001, 0.5), 3)
            individual[pos_to_mutate] = randint(
                min(individual), max(individual))

    return parents
