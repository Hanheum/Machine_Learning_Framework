import numpy as np

def ReLU_op(Z):    #relu operation done here.
    return np.maximum(Z, 0)

def ReLU_deriv(Z):  #derivation of relu
    arr = 1-(Z==0)
    return arr.astype(np.float32)

def ReLU():   #how to get relu function
    return ReLU_op, ReLU_deriv