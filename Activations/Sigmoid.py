import numpy as np

def Sigmoid_op(Z):    #sigmoid operation is done here.
    return 1/(1+np.exp(-Z))

def Sigmoid_deriv(Z):   #derivation of sigmoid
    return Sigmoid_op(Z)*(1-Sigmoid_op(Z))

def Sigmoid():       #how to get sigmoid function
    return Sigmoid_op, Sigmoid_deriv