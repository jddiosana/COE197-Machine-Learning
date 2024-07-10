import numpy as np
from tinygrad.tensor import Tensor

def mean(data):
    # get mean of the data
    return sum(data)/len(data)

def std(data):
    # get standard deviation of the data
    mu = mean(data)
    return np.sqrt(sum([(x-mu)**2 for x in data])/(len(data)-1))

def standardize(data):
    # standardize the data using the mean and standard deviation
    mu = mean(data)
    sigma = std(data)
    return [(x-mu)/sigma for x in data]

def return_to_original(data, mu, sigma):
    # return the standardized data to its original form
    return [x*sigma+mu for x in data]

def rescale_wandb(wandb, x_mu, x_sigma, y_mu, y_sigma, degree):
    # rescale the weights and bias so that they can be used to predict the originally scaled data   
    weights_rescaled = np.zeros(degree)
    w = wandb[0].data
    bias_rescaled = wandb[1].data[0]

    for i in range(1, degree+1):
        for j in range(i):
            nth_term = (w[degree-i]/x_sigma**i) * (np.math.factorial(i)/(np.math.factorial(i-j)*np.math.factorial(j))) * x_mu**(j) * (-1)**(j)
            weights_rescaled[i-j-1] += nth_term
        bias_rescaled += x_mu**i * (-1)**(i) * (w[degree-i]/x_sigma**i)

    weights_rescaled = np.array(weights_rescaled*y_sigma)
    weights_rescaled = weights_rescaled[::-1]
    bias_rescaled = (bias_rescaled*y_sigma) + y_mu

    return weights_rescaled, bias_rescaled