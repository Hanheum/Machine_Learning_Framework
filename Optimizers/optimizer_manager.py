from Optimizers.GD import gd

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