from Model.Model import Model
from Layer.Dense import Dense
from Layer.Conv2D import Conv2D
from Layer.Flatten import Flatten
from Layer.Dropout import Dropout
from Layer.AvgPool import AvgPool  
from PIL import Image
from os import listdir
import numpy as np

dataset_dir = 'C:\\Users\\chh36\\Desktop\\catdog\\'
train_dir = listdir(dataset_dir)

X = []
Y = []

img_limit = 1000

for i, category in enumerate(train_dir):
    label = np.zeros(len(train_dir))
    label[i] = 1

    titles = listdir(dataset_dir+category)
    for a, title in enumerate(titles):
        if a < img_limit:
            img = Image.open(dataset_dir+category+'\\'+title).convert('RGB')
            img = img.resize((32, 32))
            img = np.array(img)
            img = np.reshape(img, (1024, 3))
            img = img.T
            img = np.reshape(img, (3, 32, 32))
            X.append(img)
            Y.append(label)

        else:
            break

X = np.asarray(X).astype(np.float32)/255.0
Y = np.asarray(Y).astype(np.float32)

p = np.random.permutation(len(X))
X = X[p]
Y = Y[p]

print("loading completed")

network = [
    Conv2D(filters=5, kernel_size=(3, 3), input_shape=(3, 32, 32), activation='relu'),
    AvgPool((2, 2)),
    Conv2D(filters=5, kernel_size=(3, 3), activation='relu'),
    AvgPool((2, 2)),
    Flatten(),
    Dense(nods=30, activation='relu'),
    Dense(nods=30, activation='relu'),
    Dense(nods=2, activation='softmax')
]

model = Model(layers=network, optimizer='gd', loss='cce', learning_rate=0.00001)

model.accuracy(X, Y)

for i in range(10):
    model.fit(X, Y, 20)
    model.accuracy(X, Y)