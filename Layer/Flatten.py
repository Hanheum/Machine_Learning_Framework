from Layer.Default import layer
import numpy as np

class Flatten(layer):
    def __init__(self, input_shape=None):
        super().__init__(input_shape=input_shape)

        self.output_shape = None

        self.Class = 'Flatten'

    def forward(self, X):  #change shape to output shape
        self.X = X
        self.Z = np.reshape(self.X, (self.X.shape[0], *self.output_shape))
        self.A = self.Z
        return self.Z, self.A
    
    def backward(self, dZ1):   #change shape back to original shape
        dX = np.reshape(dZ1, (dZ1.shape[0], *self.input_shape))
        return dX.astype(np.float32)
    
    def __call__(self, X):
        self.X = X.astype(np.float32)
        Z, A = self.forward(self.X)
        return Z, A
    
    def optimize(self, optimizer, learning_rate):  #every function with "pass" is just formal. just to not cause any error in model class.
        pass

    def calculate_output_shape(self):
        self.output_shape = (np.prod(self.input_shape), )

    def make_variables(self):
        pass

    def reset_gradient(self):
        pass