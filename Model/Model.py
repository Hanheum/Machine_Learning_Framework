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
            
        
    def forward(self, X):
        self.X = X
        self.A = self.X
        for i, layer in enumerate(self.layers):
            self.Z, self.A = layer.forward(self.A)
            
        return self.A
    
    def backward(self, dL_first):
        self.dZ1 = dL_first
        for i in range(len(self.layers)-1, -1, -1):
            self.dZ1 = self.layers[i].backward(self.dZ1)

    def optimize(self):
        for i in range(len(self.layers)):
            self.layers[i].optimize(self.optimizer, self.learning_rate)

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

    def fit(self, X, Y, epochs, print_log=True):  #add batch system over here. overflow occurs.
        for epoch in range(epochs):
            start_time = time()
            Y_pred = self.forward(X)
            loss = self.loss(Y, Y_pred)
            dL = self.loss_deriv(Y, Y_pred)
            self.backward(dL)
            self.optimize()

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