from Model.Model import Model
from Layer.Dense import Dense
from Layer.Conv2D import Conv2D
from Layer.Flatten import Flatten
from Layer.Dropout import Dropout
from Layer.AvgPool import AvgPool    #import layers and model class

import numpy as np  
from PIL import Image
from os import listdir        #modules for importing data

X = []
Y = []

dataset_dir = 'C:\\Users\\chh36\\Desktop\\mini_mnist_things\\mini_mnist\\'  #0 and 1 from mnist
train_dir = listdir(dataset_dir)

for i, a in enumerate(train_dir):
    count = 0
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
        #total 200 imgs limitation here. you can delete the code below to un-limit the count.
        count += 1
        if count > 100:
            break

X = np.asarray(X)/255.   #data shape: (how many imgs, 1, 28, 28) 28 means width, height. 1 means color channel.
Y = np.asarray(Y)   #data shape: (how many imgs, 2) 2 categories

p = np.random.permutation(len(X))  #random shuffle data. 
X = X[p]
Y = Y[p]

layers = [
    Conv2D(filters=5, kernel_size=(3, 3), input_shape=(1, 28, 28), activation='relu'),  #convolution 2D layer
    AvgPool((2, 2)),  #average pooling 2D layer
    Conv2D(filters=5, kernel_size=(3, 3), activation='relu'),  #convolution 2D layer
    AvgPool((2, 2)),  #average pooling 2D layer
    Flatten(),   #flattening layer. almost same as reshape layer
    Dense(nods=10, activation='relu'),  #dense layer. also called NN.
    Dropout(ratio=0.2),   #dropout layer, ignoring 20% of variables.
    Dense(nods=2, activation='softmax')  #two nods(=neurons) means the output shape of this model.
]

model = Model(layers=layers, optimizer='gd', loss='cce', learning_rate=0.00001)  #make model class

model.fit(X, Y, 30, batch_size=128)   #fit the model with data X, Y. epoch 30. batch size 128
model.accuracy(X, Y)      #calculate accuracy of model using data X and Y(training data)