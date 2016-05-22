import numpy as np
from mnist_cnn import Model

img_rows = 28
img_cols = 28


def predict_test(model):
    csv = np.genfromtxt('./data/validate2.txt', delimiter=",")
    X_val = csv[:, 1:785]
    Y_val = csv[:, 0]

    X_val = X_val.reshape(X_val.shape[0], 1, img_rows, img_cols)

    print(model.predict_classes(X_val, verbose=True))

model = Model()
model.load_weights("mnist_cnn_weights.txt")
predict_test(model)