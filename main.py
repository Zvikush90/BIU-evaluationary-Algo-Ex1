import model
import os
import ga
from ga import Population, Chromosome
import logging
import sys
from time import gmtime, strftime


def git_handling(newpath):
    # git commit and push
    from git import Repo

    join = os.path.join

    # rorepo is a Repo instance pointing to the git-python repository.
    # For all you know, the first argument to Repo is a path to the repository
    # you want to work with
    repo = Repo(os.getcwd())
    assert not repo.bare
    from git import Actor

    index = repo.index  # The index contains all blobs in a flat list
    assert len(list(index.iter_blobs())) == len([o for o in repo.head.commit.tree.traverse() if o.type == 'blob'])
    # Access blob objects
    for (path, stage), entry in index.entries.items():
        pass
    new_file_path = os.path.join(repo.working_tree_dir, newpath[2:])
    main_file_path = os.path.join(repo.working_tree_dir, "main.py")
    ga_file_path = os.path.join(repo.working_tree_dir, "ga.py")
    model_file_path = os.path.join(repo.working_tree_dir, "model.py")

    index.add([new_file_path, main_file_path, ga_file_path, model_file_path])  # add a new file to the index
    # index.remove(['LICENSE'])  # remove an existing one
    # assert os.path.isfile(os.path.join(repo.working_tree_dir, 'LICENSE'))  # working tree is untouched

    author = Actor("Zvikush90", "zvikush90@gmail.com.com")
    committer = author
    # commit by commit message and author and committer
    index.commit("Auto Commit Run " + strftime("%Y-%m-%d %H:%M:%S", gmtime()), author=author, committer=committer)


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """

    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass

def run_ga(newpath):
    model.OUTPUT_PATH = newpath
    ga.OUTPUT_PATH = newpath

    p_count = 50  # 50
    gen_count = 50 #50

    pop = Population(p_count)
    pop.train_pop()
    pop.sort_pop()
    pop.save_population()
    pop.set_grade(-1.0)
    for i in xrange(gen_count):
        pop.print_gen()
        pop.evolve()
        pop.train_pop()
        pop.sort_pop()
        pop.save_population()
        pop.set_grade(-1.0)
    print "=========================================GEN GRADE HISTORY========================================="
    print pop.get_fit_history()


if __name__ == '__main__':
    # creating folder for files
    newpath = "./output_" + strftime("%Y-%m-%d %H:%M:%S", gmtime()).replace(" ", "t").replace("-", "_").replace(
        ":", "_")
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    # for ease of control to console or file
    if (False):
        # setup logging
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s:%(levelname)s:%(name)s:%(message)s',
            filename=newpath + "/out.log",
            filemode='a'
        )

        stdout_logger = logging.getLogger('STDOUT')
        sl = StreamToLogger(stdout_logger, logging.INFO)
        sys.stdout = sl

        stderr_logger = logging.getLogger('STDERR')
        sl = StreamToLogger(stderr_logger, logging.ERROR)
        sys.stderr = sl

    run_ga(newpath)
    git_handling(newpath)

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
