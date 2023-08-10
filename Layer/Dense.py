from Layer.Default import layer
from Activations.activation_manager import activation_manager
import numpy as np

class Dense(layer):
    def __init__(self, nods, input_shape=None, activation=None, use_bias=True):
        super().__init__(input_shape=input_shape)

        self.activation, self.activation_deriv = activation_manager(activation)  #activation function
        self.nods = nods  #how many nods
        self.use_bias = use_bias  #bias using option

        self.output_shape = (self.nods, )

        self.W = None
        self.B = None
        self.dW = None
        self.dB = None

    def forward(self, X):          #forward propagation
        self.X = X.astype(np.float32)
        self.Z = np.matmul(self.X, self.W)
        if self.use_bias: self.Z += self.B
        self.A = self.activation(self.Z)
        return self.Z.astype(np.float32), self.A.astype(np.float32)
    
    def backward(self, dZ1):       #backward propagation
        #dZ1 = dZ1 * self.activation_deriv(self.Z)
        self.dW = np.matmul(self.X.T, dZ1).astype(np.float32)
        self.dB = np.sum(dZ1, axis=0).astype(np.float32)
        dX = np.matmul(dZ1, self.W.T)
        return dX.astype(np.float32)
    
    def __call__(self, X):         #call forward propagation when called
        self.X = X.astype(np.float32)
        Z, A = self.forward(self.X)
        return Z, A
    
    def optimize(self, optimizer, learning_rate):    #optimize variables using optimizer
        self.W = optimizer(self.W, self.dW, learning_rate)
        self.B = optimizer(self.B, self.dB, learning_rate)

    def calculate_output_shape(self):    #calculate output shape separately, just in case you can't have output shape alone in layer. (when shapes of other layers are needed.)
        self.output_shape = (self.nods, )

    def make_variables(self):        #make variables separately, when there's no input shape on offset.
        self.W = np.random.randn(self.input_shape[0], self.nods).astype(np.float32)  #Generate Weight
        if self.use_bias: self.B = np.zeros((self.nods)).astype(np.float32)    #Generate Bias

        self.dW = np.zeros_like(self.W).astype(np.float32)
        self.dB = np.zeros_like(self.B).astype(np.float32)

    def reset_gradient(self):      #reset gradient every epoch iteration.
        self.dW = np.zeros_like(self.W).astype(np.float32)
        self.dB = np.zeros_like(self.B).astype(np.float32)