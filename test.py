from Model.Model import Model
from Layer.Dense import Dense
from Layer.Conv2D import Conv2D
import numpy as np
from PIL import Image
from os import listdir

X = []
Y = []

dataset_dir = 'C:\\Users\\chh36\\Desktop\\mini_mnist_things\\mini_mnist\\'
train_dir = listdir(dataset_dir)

for i, a in enumerate(train_dir):
    imgs_dir = dataset_dir+a
    titles = listdir(imgs_dir)
    label = np.zeros([len(train_dir)])
    label[i] = 1
    for title in titles:
        img = Image.open(dataset_dir+a+'\\'+title).convert('L')
        img = np.array(img).astype(np.float32)
        img = np.reshape(img, (1, 28, 28))
        X.append(img)
        Y.append(label)

X = np.asarray(X)
Y = np.asarray(Y)

print(X.shape)
print(Y.shape)

'''x = [
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
y = np.array(y).astype(np.float32)'''

layers = [
    Conv2D(filters=5, kernel_size=(3, 3), input_shape=(1, 28, 28), activation='relu'),
    Conv2D(filters=5, kernel_size=(3, 3), activation='relu')
]

model = Model(layers=layers, optimizer='gd', loss='cce', learning_rate=0.0001)

#model.fit(x, y, 1000)

result = model(X)
print(result.shape)