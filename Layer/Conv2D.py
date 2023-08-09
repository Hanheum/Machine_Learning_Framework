from Layer.Default import layer
from Activations.activation_manager import activation_manager
import numpy as np
from scipy.signal import correlate, convolve

class Conv2D(layer):
    def __init__(self, filters, kernel_size, input_shape=None, activation=None, use_bias=True):
        super().__init__(input_shape=input_shape)

        self.activation, self.activation_deriv = activation_manager(activation)
        self.filters = filters
        self.use_bias = use_bias

        self.output_shape = None

        self.kernel_size = kernel_size

        self.K = None
        self.B = None
        self.dK = None
        self.dB = None

    def forward(self, X):
        self.X = X.astype(np.float32)
        self.Z = np.zeros((self.X.shape[0], *self.output_shape))
        for a in range(len(self.X)):
            for i in range(self.filters):
                for j in range(self.input_shape[0]):
                    self.Z[a][i] += correlate(self.X[a][j], self.K[i][j], mode='valid', method='direct').astype(np.float32)
                self.Z[a][i] += self.B[i]
        self.A = self.activation(self.Z)
        return self.Z.astype(np.float32), self.A.astype(np.float32)

    def backward(self, dZ1):
        dZ1 = dZ1 * self.activation_deriv(self.Z)
        dZ1 = dZ1.astype(np.float32)
        m = len(self.X)
        self.dK = np.zeros_like(self.K).astype(np.float32)
        dX = np.zeros_like(self.X)
        for a, one_x in enumerate(self.X):
            for i in range(self.filters):
                for j in range(self.input_shape[0]):
                    self.dK[i][j] += correlate(one_x[j], dZ1[a][i], mode='valid', method='direct').astype(np.float32)
                    dX[a][j] = convolve(dZ1[a][i], self.K[i][j], mode='full', method='direct').astype(np.float32)
        self.dK = self.dK/m
        self.dK = self.dK.astype(np.float32)
        self.dB = sum(dZ1)/m
        self.dB = self.dB.astype(np.float32)
        return dX.astype(np.float32)

    def __call__(self, X):
        self.X = X.astype(np.float32)
        Z, A = self.forward(self.X)
        return Z, A
    
    def optimize(self, optimize_function, learning_rate):
        self.K = optimize_function(self.K, self.dK, learning_rate)
        self.B = optimize_function(self.B, self.dB, learning_rate)

    def calculate_output_shape(self):
        self.output_shape = (self.filters, self.input_shape[-1]-self.kernel_size[0]+1, self.input_shape[-1]-self.kernel_size[0]+1)

    def make_variables(self):
        self.K = np.random.randn(self.filters, self.input_shape[0], *self.kernel_size).astype(np.float32)
        if self.use_bias: self.B = np.zeros(self.output_shape).astype(np.float32)

        self.dK = np.zeros_like(self.K).astype(np.float32)
        self.dB = np.zeros_like(self.B).astype(np.float32)

    def reset_gradient(self):
        self.dK = np.zeros_like(self.K).astype(np.float32)
        self.dB = np.zeros_like(self.B).astype(np.float32)