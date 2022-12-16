import numpy as np
from tinygrad.tensor import Tensor

def predict_data(wandb, x, degree):
    # predict the output given the weights and bias
    w = wandb[0].data
    b = wandb[1].data
    y_pred = np.zeros(len(x))
    for i in range(len(x)):
        for j in range(degree):
            y_pred[i] += w[j]*x[i]**(degree-j)
        y_pred[i] += b
    return y_pred

def loss_fn(y_pred, y_true):
    # calculate the loss
    return ((y_pred - y_true)**2).mean()

def calc_grad(x, y, w, b, degree):
    # calculate the gradient of the loss function
    dl_wb = np.zeros(degree+1)
    y_pred = predict_data([w, b], x, degree)
    for i in range(degree):
        dl_wb[i] = (2*(y_pred-y)*x**(degree-i)).mean()
    dl_wb[degree] = (2*(y_pred-y)).mean()
    return dl_wb