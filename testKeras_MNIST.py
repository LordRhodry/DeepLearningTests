# -*- coding: utf-8 -*-
"""
Created on Sat May 27 18:35:05 2017

@author: Julien
"""

import numpy as np
np.random.seed(123)  # for reproducibility

from keras.models import Sequential
	
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils

#%%
from keras.datasets import mnist
 
# Load pre-shuffled MNIST data into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#%%

2
print (X_train.shape)
# (60000, 28, 28)
#%%

2
from matplotlib import pyplot as plt
plt.imshow(X_train[0])

#%%

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
print (X_train.shape)
#%%

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
#%%
print (y_train.shape)
#%%
print (y_train[:10])
#%%
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)
print (Y_train[0])

#%%
print (Y_train.shape)
#%%
model = Sequential()
model.add(Convolution2D(32, (3, 3), input_shape=(28,28,1), activation='relu'))
print (model.output_shape)

#%%
model.add(Convolution2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
print (model.output_shape)
#%%
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
#%%
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
#%%

model.fit(X_train, Y_train, 
          batch_size=32, nb_epoch=10, verbose=1)
#%%
score = model.evaluate(X_test, Y_test, verbose=0)
print(score)

#%%
print(model.metrics_names)
