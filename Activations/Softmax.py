import numpy as np

def Softmax_op(Z):      #softmax operation done here.
    y_all = np.zeros_like(Z)
    for i, a in enumerate(Z):
        c = np.max(a)
        exp_a = np.exp(a-c)
        sum_exp_a = np.sum(exp_a)
        y = exp_a/sum_exp_a
        y_all[i] = y
    return y_all

def Softmax_deriv(Z):   #softmax derivation
    softmaxed = Softmax_op(Z) 
    return (1-softmaxed)*softmaxed

def Softmax():    #how to get softmax function
    return Softmax_op, Softmax_deriv