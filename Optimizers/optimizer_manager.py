import numpy as np

def gd(target, gradient, learning_rate):  #basic gradient descent
    target = target - learning_rate * gradient
    target = target.astype(np.float32)
    return target

str_type = type("hello world")
none_type = type(None)          #variable types to handle inputs.
function_type = type(gd)

def optimizer_manager(optimizer):
    Type = type(optimizer)
    if Type == str_type:
        if optimizer == 'gd':
            return gd
        
    elif Type == none_type:
        return gd
    
    elif Type == function_type:
        return optimizer