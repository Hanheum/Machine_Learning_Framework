import numpy as np

def MSE_op(Y, Y_pred):
    size = Y.size
    dif = Y-Y_pred
    dif = dif*dif
    L = np.sum(dif)/size
    return L

def MSE_deriv(Y, Y_pred):
    m = Y.shape[1]
    dL = (2/m)*(Y_pred - Y)
    return dL

def MSE():
    return MSE_op, MSE_deriv