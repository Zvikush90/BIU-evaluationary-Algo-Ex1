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

OUTPUT_PATH = ""

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

lr_min = 0.1
lr_max = 0.7
nb_filters_min = 1  # 5
nb_filters_max = 1  # 40
nb_epochs_min = 1  # 1
nb_epochs_max = 10  # 10
dpr_min = 0.0
dpr_max = 0.5

gen_number = 0
chrom_id = 0


class Chromosome:
    'Create a member of the population.'

    # learning rate, nb filters, nb epochs, dropout rate
    def __init__(self, params=[]):
        if (len(params) == 0):
            self.lr = round(random.uniform(lr_min, lr_max), 1)
            self.nb_filters = randint(nb_filters_min, nb_filters_max)
            self.nb_epochs = randint(nb_epochs_min, nb_epochs_max)
            self.drop_rate = round(random.uniform(dpr_min, dpr_max), 1)
        else:
            self.lr = params[0]
            self.nb_filters = params[1]
            self.nb_epochs = params[2]
            self.drop_rate = params[3]
        # id related
        self.generation = gen_number
        global chrom_id
        self.id = chrom_id
        chrom_id = chrom_id + 1
        self.fitness = -1.0
        # model
        self.model = None
        self.bool_trained = False

    def get_train_status(self):
        return self.bool_trained

    def train(self):
        print "Current Individual: " + str(self.generation) + "_" + str(self.id)
        print "Learning rate - " + str(self.lr) + ", Nb filters - " + str(self.nb_filters) + ", Nb epochs - " \
              + str(self.nb_epochs) + ", Dropout rate - " + str(self.drop_rate)
        self.model = CNN(nb_classes, self.nb_filters, self.nb_epochs, self.drop_rate, img_rows, img_cols)
        self.history = self.model.train(self.lr, X_train, Y_train)
        # cnn_model.load_model_from_file(1)
        score = self.model.test(str(gen_number) + "_" + str(self.id), X_test, Y_test)
        self.fitness = score[1]
        self.model.write_model_to_file(str(self.generation) + "_" + str(self.id))
        self.bool_trained = True

    def set_fitness(self, num):
        self.fitness = num

    def get_fitness(self):
        return self.fitness

    def get_id(self):
        return self.id

    def get_params(self):
        return [self.lr, self.nb_filters, self.nb_epochs, self.drop_rate]


class Population:
    def __init__(self, count):
        self.count = count
        self.pop_list = [Chromosome() for i in xrange(count)]
        self.grade = -1.0
        self.fitness_history = []

    def print_gen(self):
        print "=========================================GEN " + str(
            gen_number) + "========================================="

    def train_pop(self):
        for ind in self.pop_list:
            if (ind.get_train_status() == False):
                ind.train()

    def all_trained(self):
        for ind in self.pop_list:
            if (ind.get_train_status() == False):
                return False
        return True

    def sort_pop(self):
        self.pop_list.sort(key=lambda x: x.fitness, reverse=True)

    def get_grade(self):
        if self.grade == -1.0:
            num = [x.fitness for x in self.pop_list]
            summed = sum(num)
            self.grade = summed / (self.count * 1.0)
            print "GEN GRADE = " + str(self.grade)
            self.fitness_history.append(self.grade)

        return self.grade

    def get_fit_history(self):
        return self.fitness_history

    def save_population(self):
        # writing to file
        text_file = open(OUTPUT_PATH + "/GEN_" + str(gen_number) + ".txt", "w")
        arr = [(self.pop_list[i].get_id(), self.pop_list[i].get_params(), self.pop_list[i].get_fitness()) for i in
               xrange(self.count)]
        text_file.write(
            str(self.get_grade()) + "\n" + "ID, [Learning rate, Nb filters, Nb epochs, Dropout rate], Fitness\n" + str(
                arr))
        text_file.close()

    def roulette_select(self, num_select):
        fit_list = [x.fitness for x in self.pop_list]
        summed = float(sum(fit_list))

        rel_fitness = [f / summed for f in fit_list]
        # Generate probability intervals for each individual
        probs = [sum(rel_fitness[:i + 1]) for i in range(len(rel_fitness))]
        # Draw new population
        new_population = []
        for n in xrange(num_select):
            from random import random
            r = random()
            for (i, individual) in enumerate(self.pop_list):
                if r <= probs[i]:
                    new_population.append(individual)
                    break
        return new_population

    def evolve(self, retain=0.2, mutate=0.01):
        self.sort_pop()
        parents_length = int(self.count * retain)

        # randomly add other individuals to
        # promote genetic diversity
        parents = self.roulette_select(parents_length)

        # advance gen
        global gen_number
        gen_number = gen_number + 1

        # crossover parents to create children
        num_to_add = self.count - len(parents)
        children = []
        while len(children) < num_to_add:
            male = randint(0, len(parents) - 1)
            female = randint(0, len(parents) - 1)
            # if male != female:
            male = parents[male].get_params()
            female = parents[female].get_params()
            index = randint(0,len(male)-1)
            child1 = male[:index] + female[index:]
            child2 = female[:index] + male[index:]
            children.append(Chromosome(child1))
            children.append(Chromosome(child2))
        parents.extend(children)

        # mutate some individuals
        for individual in parents:
            from random import random
            if mutate > random():
                pos_to_mutate = randint(0, len(individual) - 1)
                if (pos_to_mutate == 0):
                    round(random.uniform(lr_min, lr_max), 3)
                elif (pos_to_mutate == 1):
                    randint(nb_filters_min, nb_filters_max)
                elif (pos_to_mutate == 2):
                    randint(nb_epochs_min, nb_epochs_max)
                else:
                    round(random.uniform(dpr_min, dpr_max), 1)
                individual[pos_to_mutate] = randint(
                    min(individual), max(individual))
