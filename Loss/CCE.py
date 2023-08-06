import numpy as np

def CCE_op(Y, Y_pred):
    Y_pred = np.clip(Y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(Y * np.log(Y_pred + 1e-7) + (1 - Y) * np.log(1 - Y_pred + 1e-7))

def CCE_deriv(Y, Y_pred):
    Y_pred = np.clip(Y_pred, 1e-7, 1 - 1e-7)
    cost = ((1 - Y) / (1 - Y_pred + 1e-7) - Y / (Y_pred + 1e-7)) / np.size(Y)
    return cost * 1000

def CCE():
    return CCE_op, CCE_deriv