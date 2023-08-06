from Activations.ReLU import ReLU
from Activations.Softmax import Softmax
from Activations.Sigmoid import Sigmoid

def no_effect(Z):   #function that has no effect, used when no activation designated
    return Z

def no_effect_deriv(Z):  #derivation of no effect, value is 1 every time
    return 1

str_type = type("hello world")    
none_type = type(None)                #types of variables, used for analyzing input
function_type = type(no_effect)

def activation_manager(activation):
    if type(activation) == str_type:
        if activation == 'relu':      #ReLU
            relu, relu_deriv = ReLU()
            return relu, relu_deriv
        
        elif activation == 'softmax':   #Softmax
            softmax, softmax_deriv = Softmax()
            return softmax, softmax_deriv
        
        elif activation == 'sigmoid':   #Sigmoid
            sigmoid, sigmoid_deriv = Sigmoid()
            return sigmoid, sigmoid_deriv
    
    elif type(activation) == none_type:  #when there's no activation
        return no_effect, no_effect_deriv
     
    elif type(activation) == function_type:  #when input is function
        op, deriv = activation()
        return op, deriv