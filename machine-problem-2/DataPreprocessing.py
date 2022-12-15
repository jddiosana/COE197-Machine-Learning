import numpy as np

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

def rescale_wandb(wandb, x_mu, x_sigma, y_mu, y_sigma):
    # rescale the weights and bias so that they can be used to predict the originally scaled data
    w = wandb[0]
    b = wandb[1]
    w0 = w.data[0] * y_sigma/x_sigma**4
    w1 = y_sigma * ((w.data[1]/x_sigma**3) - (4*x_mu*w.data[0]/x_sigma**4))
    w2 = y_sigma * ((w.data[2]/x_sigma**2) - (3*x_mu*w[1]/x_sigma**3) + (6*x_mu**2*w[0]/x_sigma**4))
    w3 = y_sigma * ((w.data[3]/x_sigma) - (2*x_mu*w.data[2]/x_sigma**2) + (3*x_mu**2*w.data[1]/x_sigma**3) - (4*x_mu**3*w.data[0]/x_sigma**4))
    b = y_sigma*(b.data - (w.data[3]*x_mu/x_sigma) + (w.data[2]*x_mu**2/x_sigma**2) - (w.data[1]*x_mu**3/x_sigma**3) + (w.data[0]*x_mu**4/x_sigma**4)) + y_mu

    return [w0, w1, w2, w3], b