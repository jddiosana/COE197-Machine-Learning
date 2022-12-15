import numpy as np
from tinygrad.tensor import Tensor

def split_indices(data, k):
    # split the data into k folds where the first fold is data with indices 0, k, 2k, ..., and so on
    folds = []
    for i in range(k):
        fold = []
        for j in range(len(data)):
            if j%k == i:
                fold.append(j)
        fold = np.array(fold)
        folds.append(fold)
    return folds

def split_data(x_data, y_data, fold):

    x_copy = x_data.copy()
    y_copy = y_data.copy()
    x_train = np.delete(x_copy, fold) # delete the fold from the data to be used for training
    y_train = np.delete(y_copy, fold)
    x_validation = x_copy[fold] # use the fold for validation
    y_validation = y_copy[fold]

    x_train = Tensor(x_train)
    y_train = Tensor(y_train)
    x_validation = Tensor(x_validation)
    y_validation = Tensor(y_validation)

    return x_train, y_train, x_validation, y_validation
    