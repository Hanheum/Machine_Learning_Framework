from Model.Model import Model
from Layer.Dense import Dense
import numpy as np

x = [
    [0, 0],
    [1, 0],
    [1, 1],
    [0, 0],
    [0, 0],
    [0, 1]
]

y = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0],
    [1, 0, 0],
    [0, 0, 1]
]

x = np.array(x).astype(np.float32)
y = np.array(y).astype(np.float32)

layers = [
    Dense(nods=5, input_shape=(2, ), activation='relu'),
    Dense(nods=5, activation='relu'),
    Dense(nods=3, activation='softmax'),
]

model = Model(layers=layers, optimizer='gd', loss='cce', learning_rate=0.0001)

model.fit(x, y, 1000)

result = model(x)
print(np.round(result, 3))