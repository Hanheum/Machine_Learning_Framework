from math import floor

nonetype = type(None)

def gd_op(target, gradient, learning_rate):  #actual gradient descent operation done here.
    return target - learning_rate*gradient

def gd(X, Y, layers, forward, backward, loss, loss_deriv, learning_rate, batch_size=None): #everything you need to train model.
    if type(batch_size) == nonetype:
        batch_size = len(X)

    iters = floor(len(X)/batch_size)

    for i in range(iters):
        x = X[batch_size*i:batch_size*(i+1)]
        y = Y[batch_size*i:batch_size*(i+1)]
        y_pred = forward(x)
        dL = loss_deriv(y, y_pred)
        backward(dL)

    Y_pred = forward(X)
    loss_value = loss(Y, Y_pred)

    for i in range(len(layers)):  #apply optimization here. 
        layers[i].optimize(gd_op, learning_rate)

    return loss_value