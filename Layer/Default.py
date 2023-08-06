class layer:
    def __init__(self, input_shape=None, name=None):
        self.name = name
        self.X = None   #input
        self.Z = None   #result before activation
        self.A = None   #result after activation
        self.input_shape = input_shape
        self.output_shape = None

    def forward(self, X):
        self.X = X
        self.Z = self.X
        self.A = self.Z
        #insert operations here

        return self.Z, self.A

    def backward(self, X, next_Z_deriv):
        self.X = X
        dZ = next_Z_deriv

        #insert operations here

        return dZ
    
    def __call__(self, X):
        self.X = X
        Z, A = self.forward(self.X)
        return Z, A