from Optimizers.optimizer_manager import optimizer_manager
from Loss.loss_manager import loss_manager
from time import time
import numpy as np

class Model:
    def __init__(self, layers=[], optimizer='gd', loss='mse', learning_rate=0.01):
        self.layers = layers     #entire layers stacked, very complex layers can be added here
        self.X = None
        self.Y = None
        self.Zs = None
        self.As = None
        self.dZ1 = None
        self.optimizer = optimizer_manager(optimizer)
        self.loss, self.loss_deriv = loss_manager(loss)()
        self.learning_rate = learning_rate

        self.layers[0].calculate_output_shape()
        self.layers[0].make_variables()
        for i in range(len(self.layers)-1):
            self.layers[i+1].input_shape = self.layers[i].output_shape
            self.layers[i+1].calculate_output_shape()
            self.layers[i+1].make_variables()
            
        
    def forward(self, X):  #forward propagation
        self.X = X
        self.A = self.X
        for layer in self.layers:
            self.Z, self.A = layer.forward(self.A)
            
        return self.A
    
    def backward(self, dL_first):  #backward propagation
        self.dZ1 = dL_first
        for i in range(len(self.layers)-1, -1, -1):
            self.dZ1 = self.layers[i].backward(self.dZ1)

    def reset_gradient(self):    #reset gradients every epoch iteration.
        for i in range(len(self.layers)):
            self.layers[i].reset_gradient()

    def set_learning_rate(self, learning_rate):   #option to change learning rate
        self.learning_rate = learning_rate

    def fit(self, X, Y, epochs, print_log=True, batch_size=None):
        for epoch in range(epochs):
            start_time = time()
            self.reset_gradient()
            loss = self.optimizer(X, Y, self.layers, self.forward, self.backward, self.loss, self.loss_deriv, self.learning_rate, batch_size=batch_size)

            end_time = time()

            if print_log:
                duration = end_time-start_time
                duration = round(duration, 3)
                log = 'epoch: {} | loss: {} | duration: {} seconds'.format(epoch+1, loss, duration)
                print(log)

    def accuracy(self, X, Y):
        result = self.forward(X)
        result = list(map(np.argmax, result))
        result = np.array(result)
        answers = list(map(np.argmax, Y))
        answers = np.array(answers)
        dif = result-answers
        corrects = dif == 0
        corrects = np.sum(corrects)
        Accuracy = 100*corrects/len(X)
        print(Accuracy, '%')
        return Accuracy

    def __call__(self, X):
        return self.forward(X)