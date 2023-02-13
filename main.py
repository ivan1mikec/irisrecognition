from pickle import FALSE
from wsgiref import validate
import numpy as np
import pandas as pd

import csv
import cv2
import os, glob
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import time
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from matplotlib import pyplot as plt
from keras.utils import np_utils 
import os
from keras_preprocessing import image
import numpy as np
import random

img_data_lista = []
label_lista = []
img_data_lista1 = []
img_data_lista2 = []
img_data_lista3 = []
img_data_lista4 = []
img_data_lista5 = []
label_lista1 = []
label_lista2 = []
label_lista3 = []
label_lista4 = []
label_lista5 = []


#podjela normaliziranih slika u 5 foldera
for file in os.listdir("Normalized_Images/"):
    if file.endswith(".bmp"):
        img_path = "Normalized_Images/" + file
        img = image.load_img(img_path, target_size =(200,150))
        img_tensor = image.img_to_array(img)
        img_data_lista.append(img_tensor)
        label_lista.append(file[0:3])
        if file[4:5]=='1':
            img_data_lista1.append(img_tensor)
            label_lista1.append(file[0:3])
        if file[4:5]=='2':
            img_data_lista2.append(img_tensor)
            label_lista2.append(file[0:3])
        if file[4:5]=='3':
            img_data_lista3.append(img_tensor)
            label_lista3.append(file[0:3])
        if file[4:5]=='4':
            img_data_lista4.append(img_tensor)
            label_lista4.append(file[0:3])
        if file[4:5]=='5':
            img_data_lista5.append(img_tensor)
            label_lista5.append(file[0:3])
 
import gc
gc.collect()

#folderi 1-4 za treniranje (80%)
train_img = np.array(img_data_lista1)
train_img = np.append(train_img, np.array(img_data_lista2),axis=0)
train_img = np.append(train_img, np.array(img_data_lista3),axis=0)
train_img = np.append(train_img, np.array(img_data_lista4),axis=0)

train_img /= 255


train_label = np.array(label_lista1)
train_label = np.append(train_label, np.array(label_lista2), axis = 0)
train_label = np.append(train_label, np.array(label_lista3), axis = 0)
train_label = np.append(train_label, np.array(label_lista4), axis = 0)
#test ne koristimo
test_img = np.array(img_data_lista4)
test_img /= 255

test_label = np.array(label_lista4)
#folder 5 za validaciju (20%)
validate_img = np.array(img_data_lista5)
validate_img /= 255

validate_label = np.array(label_lista5)

print(train_label.shape)

from keras.utils import np_utils

train_label = np.array(train_label)
test_label = np.array(test_label)
train_label = np_utils.to_categorical(train_label)
test_label = np_utils.to_categorical(test_label)
validate_label = np_utils.to_categorical(validate_label)
num_classes = train_label.shape[1]
print("Podaci normalizirani i hot-coded")
print(num_classes)

#resize slika za CNN model

test_img.reshape(-1, 200, 150, 3)
train_img.reshape(-1, 200, 150, 3)
validate_img.reshape(-1, 200, 150 ,3)

def createCNNModel(num_classes):
    model = Sequential()
    model.add(Convolution2D(32,3,3, input_shape =(200,150,3), activation = 'relu'))
    model.add(Convolution2D(32,3,3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(512, activation = 'relu'))
    model.add(Dropout(0.6))
    model.add(Dense(num_classes,activation='softmax'))

    epochs = 30
    lrate = 0.01

    decay = lrate/epochs
    sgd = SGD (lr = lrate, momentum=0.9, decay = decay, nesterov=False)
    model.compile (loss ='categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])
    print(model.summary())
    return model, epochs



model, epochs = createCNNModel(num_classes)
print("CNN model stvoren")

print(validate_img.shape)

modelfit = model.fit(train_img, train_label, validation_data = (validate_img, validate_label), epochs = epochs, batch_size= 64)

train_loss = modelfit.history['loss']
val_loss = modelfit.history['val_loss']
train_acc = modelfit.history['accuracy']
val_acc = modelfit.history['val_accuracy']
xc = range(epochs)
#prikaz rezultata grafom
plt.figure(figsize=(20,10))
plt.plot(xc, train_acc, label = 'Train acc')
plt.plot(xc, val_acc, label = 'Validation acc')
plt.legend(loc ='upper left', prop = {'size':20})
plt.title("Preciznost", size = 20)
plt.xlabel("Iteracije", size = 20)
plt.ylabel("Preciznost", size = 20)
plt.show()
