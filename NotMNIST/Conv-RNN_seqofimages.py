# -*- coding: utf-8 -*-
"""
Created on Sun May 28 14:35:10 2017

@author: Julien
"""

from __future__ import print_function
import numpy as np
from six.moves import cPickle as pickle

from keras.models import Sequential
from keras.preprocessing import sequence
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, recurrent
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from six.moves import range

pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)
          
image_size = 28
num_labels = 10
num_channels = 1 # grayscale

def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


#%%
training_size = 500000
valid_size = 100000
test_size = 50000
max_sequence = 5
nb_classes = num_labels +1 # in this case 11 (adding the blank image) 

train_set = np.zeros(shape=(training_size,max_sequence, image_size, image_size, num_channels),dtype=np.float32)
train_lab = np.zeros(shape=(training_size,max_sequence, nb_classes),dtype=np.float32)

for i in range( training_size):
    leng = np.random.randint(1,max_sequence+1)
    indices  = np.random.choice(max_sequence,leng)
    targets = np.random.choice(train_dataset.shape[0],leng)
    train_set [i, indices , :,:,:] = train_dataset[targets,:,:,:]
    train_lab [i , : , -1  ] = 1
    train_lab [i , indices , :-1] = train_labels[targets, :]
    train_lab [i, indices, -1] = 0
    
#%%

# Parameters for the model and dataset
TRAINING_SIZE = 400000
INVERT = True
# Try replacing GRU, or SimpleRNN
RNN = recurrent.LSTM
HIDDEN_SIZE = 128

LAYERS = 2

#%%
print('Build model...')
model = Sequential()
model.add(TimeDistributed(Convolution2D(32, (3, 3),  activation='relu'),input_shape=(5,28,28,1),))
model.add(TimeDistributed(Convolution2D(32, (3, 3), activation='relu')))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2,2))))
model.add(TimeDistributed(Flatten()))

model.add(RNN(HIDDEN_SIZE))
# For the decoder's input, we repeat the encoded input for each time step
model.add(RepeatVector(max_sequence))
# The decoder RNN could be multiple layers stacked or a single layer
for _ in range(LAYERS):
    model.add(RNN(HIDDEN_SIZE, return_sequences=True))

# For each of step of the output sequence, decide which character should be chosen
model.add(TimeDistributed(Dense(nb_classes)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    

#%%


model.fit(train_set, train_lab, batch_size=32, epochs=25, verbose=1)


