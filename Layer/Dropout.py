from Layer.Default import layer
import numpy as np

class Dropout(layer):
    def __init__(self, ratio, input_shape=None):
        super().__init__(input_shape=input_shape)

        self.ratio = ratio
        self.count = 0
        self.count_backup = 0
        self.random_blanks = None
        self.var = None

    def forward(self, X):
        if self.training: 
            self.random_blanks = np.ones((np.prod(self.input_shape)))
            self.random_blanks[0:self.count] = np.zeros((self.count))
            np.random.shuffle(self.random_blanks)
            self.random_blanks = np.reshape(self.random_blanks, self.input_shape)
            self.var = self.random_blanks
        else: 
            self.var = 1
        self.X = X
        self.Z = self.X * self.var
        self.A = self.Z
        return self.Z, self.A

    def backward(self, dZ1):
        return dZ1 * self.random_blanks

    def __call__(self, X):
        self.X = X
        Z, A = self.forward(self.X)
        return Z, A

    def optimize(self, optimizer, learning_rate):
        pass

    def calculate_output_shape(self):
        self.output_shape = self.input_shape
        self.count = round(np.prod(self.input_shape) * self.ratio)
        
    def make_variables(self):
        pass

    def reset_gradient(self):
        pass