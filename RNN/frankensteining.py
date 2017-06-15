# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 12:57:07 2017

@author: Julien
"""

## character generation with Keras based on Frankenstein from Mary Shelley

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

#%%  load ascii text and covert to lowercase
filename = "Frankenstein.txt"
raw_text = open(filename).read()
raw_text = raw_text.lower()

#%% create mapping of unique chars to integers
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))

#%%

n_chars = len(raw_text)
n_vocab = len(chars)
print ("Total Characters: ", n_chars)
print ("Total Vocab: ", n_vocab)

#%% prepare the dataset of input to output pairs encoded as integers
seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
	seq_in = raw_text[i:i + seq_length]
	seq_out = raw_text[i + seq_length]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print ("Total Patterns: ", n_patterns)

#%% one hot encode and reshape X to be [samples, time steps, features]

X = numpy.reshape(dataX, (n_patterns, seq_length, 1))

X = X / n_vocab
# one hot encode the output variable
y = np_utils.to_categorical(dataY)

#%%
model2 = Sequential()
model2.add(LSTM(512, input_shape=(X.shape[1], X.shape[2]),return_sequences=True))
model2.add(LSTM(512,return_sequences=True))
model2.add(LSTM(512))
model2.add(Dense(y.shape[1], activation='softmax'))
model2.compile(loss='categorical_crossentropy', optimizer='adam')

#%%
# define the checkpoint
filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

#%%
# fit the model
model2.fit(X, y, epochs=1, batch_size=64, callbacks=callbacks_list)

#%%
def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

#%% use latest checkpoint weights to predict
filename = "weights-improvement-01-1.7884.hdf5"
model2.load_weights(filename)
model2.compile(loss='categorical_crossentropy', optimizer='adam')

#%%
for diversity in [0.2, 0.5, 1.0, 1.2]:
    print()
    print('----- diversity:', diversity)
    int_to_char = dict((i, c) for i, c in enumerate(chars))
    # pick a random seed
    start = np.random.randint(0, len(dataX)-1)
    pattern = dataX[start]
    print ("Seed:")
    print ("\"", ''.join([int_to_char[value] for value in pattern]), "\"")
    # generate characters
    for i in range(1000):
    	x = np.reshape(pattern, (1, len(pattern), 1))
    	x = x / float(n_vocab)
    	prediction = model2.predict(x, verbose=0)
    	index = sample(prediction[0], diversity)
    	result = int_to_char[index]
    	seq_in = [int_to_char[value] for value in pattern]
    	print(result , end='')
    	pattern.append(index)
    	pattern = pattern[1:len(pattern)]
    print ("\nDone.")