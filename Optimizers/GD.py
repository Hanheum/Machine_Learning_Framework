def gd_op(target, gradient, learning_rate):
    return target - learning_rate*gradient

def gd(X, Y, layers, forward, backward, loss, loss_deriv, learning_rate):
    Y_pred = forward(X)
    loss_value = loss(Y, Y_pred)
    dL = loss_deriv(Y, Y_pred)
    backward(dL)
    for i in range(len(layers)):
        layers[i].optimize(gd_op, learning_rate)
    return loss_value