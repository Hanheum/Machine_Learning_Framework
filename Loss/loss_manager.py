from Loss.MSE import MSE
from Loss.CCE import CCE

str_type = type("hello world")
none_type = type(None)           #various types for input management.
function_type = type(MSE)

def loss_manager(loss):
    Type = type(loss)
    if Type == str_type:
        if loss == 'mse':   #MSE: Mean Squared Error
            return MSE
        
        elif loss == 'cce':  #CCE: Categorical CrossEntropy
            return CCE
        
    elif Type == none_type:
        return MSE
    
    elif Type == function_type:
        return loss