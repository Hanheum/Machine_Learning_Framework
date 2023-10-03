from Layer.Default import layer
import numpy as np
from scipy.signal import correlate

class AvgPool(layer):
    def __init__(self, kernel_size, input_shape=None):
        super().__init__(input_shape=input_shape)

        self.kernel_size = kernel_size
        self.K = np.ones(kernel_size)

        self.Class = 'AvgPool'

    def forward(self, X):
        self.X = X
        self.Z = np.zeros((len(self.X), *self.output_shape))
        for a, one_x in enumerate(self.X):
            for i in range(one_x.shape[0]):
                self.Z[a][i] = correlate(one_x[i], self.K, mode='valid', method='direct')
        self.A = self.Z
        return self.Z.astype(np.float32), self.A.astype(np.float32)

    def backward(self, dZ1):
        dX = np.zeros_like(self.X)
        for a, one_dz_next in enumerate(dZ1):
            for i in range(self.input_shape[0]):
                dX[a][i] = correlate(one_dz_next[i], self.K, mode='full', method='direct')
        return dX.astype(np.float32)
    
    def __call__(self, X):
        self.X = X.astype(np.float32)
        Z, A = self.forward(self.X)
        return Z, A
    
    def optimize(self, optimize_function, learning_rate):
        pass
    
    def calculate_output_shape(self):
        self.output_shape = (self.input_shape[0], self.input_shape[-1]-self.kernel_size[0]+1, self.input_shape[-1]-self.kernel_size[0]+1)
    
    def make_variables(self):
        pass

    def reset_gradient(self):
        pass