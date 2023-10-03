from Optimizers.optimizer_manager import optimizer_manager
from Loss.loss_manager import loss_manager
from time import time
import numpy as np
import os

from Layer.Dense import Dense
from Layer.Conv2D import Conv2D
from Layer.Reshape import Reshape
from Layer.AvgPool import AvgPool
from Layer.Flatten import Flatten

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

        for i in range(len(self.layers)):
            self.layers[i].name = f'{i}'
        
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
        for i in range(len(self.layers)):
            self.layers[i].training = True

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
        
        for i in range(len(self.layers)):
            self.layers[i].training = False

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
        print(round(Accuracy), '%')
        return Accuracy

    def __call__(self, X):
        return self.forward(X)
    
    def save_variable(self, variable, saving_dir):
        variable = variable.tobytes()
        file = open(saving_dir, 'wb')
        file.write(variable)
        file.close()
    
    def save(self, saving_dir):
        for i in range(len(self.layers)):
            Class = self.layers[i].Class
            name = self.layers[i].name

            slot = saving_dir+'\\'+name

            if not os.path.isdir(saving_dir):
                os.makedirs(saving_dir)

            if not os.path.isdir(slot):
                os.makedirs(slot)
            else:
                files = os.listdir(slot)
                if len(files) != 0:
                    for title in files:
                        os.remove(slot+'\\'+title)

            if Class == 'Dense':
                W = self.layers[i].W
                B = self.layers[i].B

                W_shape = W.shape
                B_shape = B.shape

                file = open(slot+'\\'+'shapes.txt', 'a')
                file.write(f'{W_shape}\n')
                file.write(f'{B_shape}')
                file.close()

                self.save_variable(W, slot+'\\'+'W')
                self.save_variable(B, slot+'\\'+'B')

            elif Class == 'Conv2D':
                K = self.layers[i].K
                B = self.layers[i].B

                K_shape = K.shape
                B_shape = B.shape

                file = open(slot+'\\'+'shapes.txt', 'a')
                file.write(f'{K_shape}\n')
                file.write(f'{B_shape}')
                file.close()

                self.save_variable(K, slot+'\\'+'K')
                self.save_variable(B, slot+'\\'+'B')

            else:
                pass

    def load(self, directory):
        for i in range(len(self.layers)):
            name = self.layers[i].name
            Class = self.layers[i].Class

            slot = directory+'\\'+name         

            if Class == 'Dense':
                shape_file = open(slot+'\\'+'shapes.txt', 'r').readlines()   
                W = open(slot+'\\'+'W', 'rb').read()
                B = open(slot+'\\'+'B', 'rb').read()
                
                W = np.frombuffer(W, dtype=np.float32)
                W = np.reshape(W, eval(shape_file[0]))

                B = np.frombuffer(B, dtype=np.float32)
                B = np.reshape(B, eval(shape_file[1]))

                self.layers[i].W = W
                self.layers[i].B = B

            elif Class == 'Conv2D':
                shape_file = open(slot+'\\'+'shapes.txt', 'r').readlines()   
                K = open(slot+'\\'+'K', 'rb').read()
                B = open(slot+'\\'+'B', 'rb').read()

                K = np.frombuffer(K, dtype=np.float32)
                K = np.reshape(K, eval(shape_file[0]))

                B = np.frombuffer(B, dtype=np.float32)
                B = np.reshape(B, eval(shape_file[1]))

                self.layers[i].K = K
                self.layers[i].B = B

            else:
                pass