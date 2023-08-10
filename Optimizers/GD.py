def gd_op(target, gradient, learning_rate):  #actual gradient descent operation done here.
    return target - learning_rate*gradient

def gd(X, Y, layers, forward, backward, loss, loss_deriv, learning_rate): #everything you need to train model.

    #since it's taking all training data all together, breaking data in batches can be done.

    Y_pred = forward(X)
    loss_value = loss(Y, Y_pred)
    dL = loss_deriv(Y, Y_pred)
    backward(dL)

    for i in range(len(layers)):  #apply optimization here. 
        layers[i].optimize(gd_op, learning_rate)

    return loss_value