# import libraries

from tinygrad.tensor import Tensor
from tinygrad import optim

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from DataPreprocessing import mean, std, standardize, return_to_original, rescale_wandb
from DataSplit import split_indices, split_data
from TrainModel import predict_data, loss_fn, calc_grad

import argparse

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=1000, help='number of epochs')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--k', type=int, default=5, help='number of folds')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
args = parser.parse_args()

# read training data
train = pd.read_csv('data_train.csv')

# preprocess the data
x_mean, x_std = mean(train['x']), std(train['x'])
y_mean, y_std = mean(train['y']), std(train['y'])
# standardize x and y
x = standardize(train['x'])
y = standardize(train['y'])

# convert to Tensor
x, y = Tensor(x), Tensor(y)

# set hyperparameters
epochs = args.epochs
lr = args.lr
k = args.k
batch_size = args.batch_size

val_losses = [] # store validation losses
params = [] # store parameters

# split the data into k folds
folds = split_indices(x.data, k)

# train model using k-fold cross validation
for i in range(k):
    x_train, y_train, x_validation, y_validation = split_data(x.data, y.data, folds[i])

    # initialize parameters
    w = Tensor(np.random.randn(4)) # 4 weights
    b = Tensor(np.random.randn(1)) # 1 bias term
    # initialize optimizer
    opt = optim.SGD([w, b], lr=lr)

    # train model
    for epoch in tqdm(range(epochs)):
        idx = np.random.permutation(x_train.shape[0]) # shuffle data
        batches = np.array_split(idx, len(idx)//batch_size) # split data into batches

        # train model on batches 
        for batch in batches:
            xb = x_train.data[batch]
            yb = y_train.data[batch]

            y_pred = predict_data([w, b], xb) # make predictions
            loss = loss_fn(y_pred, yb) # calculate loss
            grad = calc_grad(xb, yb, w, b) # calculate gradients
            opt.params[0].grad = np.array(grad[0:len(grad)-1]) # update gradients of weights
            opt.params[1].grad = np.array(grad[len(grad)-1]) # update gradients of bias term
            opt.step() # update parameters

    #validate model
    y_pred = predict_data([w, b], x_validation.data)
    val_loss = loss_fn(y_pred, y_validation.data)
    val_losses.append(val_loss)

    #save parameters
    params.append([w.data, b.data])

best_params = params[np.argmin(val_losses)] # get parameters with lowest validation loss

#return data to original scale
x_orig = return_to_original(x.data, x_mean, x_std)
y_orig = return_to_original(y.data, y_mean, y_std)
x_orig = Tensor(x_orig)
y_orig = Tensor(y_orig)

#rescale parameters for original data
best_params_rescaled = rescale_wandb(best_params, x_mean, x_std, y_mean, y_std)
best_params_rescaled = [Tensor(best_params_rescaled[0]), Tensor(best_params_rescaled[1])]

#make predictions on original data
y_pred_rescaled = predict_data(best_params_rescaled, x_orig.data)

# calculate rmse
rmse_train = np.sqrt(np.mean((y_pred_rescaled - y_orig.data)**2))

# open the test data
test = pd.read_csv('data_test.csv')

#create x and y test data
x_test = test['x']
y_test = test['y']

#predict y values
y_pred_test = predict_data(best_params_rescaled, x_test)

# calculate rmse
rmse_test = np.sqrt(np.mean((y_pred_test - y_test)**2))

#print the parameters
print('Weights:', best_params_rescaled[0].data)
print('Bias:', best_params_rescaled[1].data[0])
print('Polynomial equation:', 'y =', best_params_rescaled[0].data[0], '* x^3 +', best_params_rescaled[0].data[1], '* x^2 +', best_params_rescaled[0].data[2], '* x +', best_params_rescaled[0].data[3], '* x +', best_params_rescaled[1].data[0])

print('RMSE on training data:', rmse_train)
print('RMSE on test data:', rmse_test)

