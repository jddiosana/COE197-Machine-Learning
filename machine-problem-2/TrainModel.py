def predict_data(wandb, x):
    # predict the output given the weights and bias
    w = wandb[0]
    b = wandb[1]
    return w.data[0]*x**4 + w.data[1]*x**3 + w.data[2]*x**2 + w.data[3]*x + b.data

def loss_fn(y_pred, y_true):
    # calculate the loss
    return ((y_pred - y_true)**2).mean()

def calc_grad(x, y, w, b):
    # calculate the gradient of the loss function
    
    y_pred = predict_data([w, b], x)
    dL_w0 = 2*(y_pred-y)*x**4
    dL_w1 = 2*(y_pred-y)*x**3
    dL_w2 = 2*(y_pred-y)*x**2
    dL_w3 = 2*(y_pred-y)*x
    dL_b = 2*(y_pred-y)

    return [dL_w0.mean(), dL_w1.mean(), dL_w2.mean(), dL_w3.mean(), dL_b.mean()]
    