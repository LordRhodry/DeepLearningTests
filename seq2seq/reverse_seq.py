# -*- coding: utf-8 -*-
"""
Created on Wed May 31 16:49:31 2017

@author: Julien
"""
from __future__ import print_function
import numpy as np
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, recurrent
from six.moves import range

#%%

class CharacterTable(object):
    '''
    Given a set of characters:
    + Encode them to a one hot integer representation
    + Decode the one hot integer representation to their character output
    + Decode a vector of probabilities to their character output
    '''
    def __init__(self, chars, maxlen):
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
        self.maxlen = maxlen

    def encode(self, C, maxlen=None):
        maxlen = maxlen if maxlen else self.maxlen
        X = np.zeros((maxlen, len(self.chars)))
        for i, c in enumerate(C):
            X[i, self.char_indices[c]] = 1
        return X

    def decode(self, X, calc_argmax=True):
        if calc_argmax:
            X = X.argmax(axis=-1)
        return ''.join(self.indices_char[x] for x in X)

#%%

# Parameters for the model and dataset
TRAINING_SIZE = 400000
INVERT = True
# Try replacing GRU, or SimpleRNN
RNN = recurrent.LSTM
HIDDEN_SIZE = 128
BATCH_SIZE = 128
LAYERS = 2
MAXLEN = 20

chars = 'abcdefghijklmnopqrstuvwxyz '
ctable = CharacterTable(chars, MAXLEN)

#%%
# Create 2 datasets of 100,000 "words" each will be an array of length 1-26 
# containing an int 0-26 : label will be train reversed
train_set = list()
train_label = list()

for i in range(TRAINING_SIZE):
    length = np.random.randint(1,MAXLEN)
    trainval = list()
    trainlav = list()
    for j in range(length):
        val = np.random.randint(1,26) +64
        trainval.append(val)
        trainlav.insert(0, val)
    train_set.append(np.array(trainval))
    train_label.append(np.array(trainlav))
#%%
train_set = np.array(train_set)
train_label = np.array(train_label)
train_set = sequence.pad_sequences (train_set , padding = 'post', maxlen = MAXLEN)
train_label = sequence.pad_sequences (train_label, padding = 'post' , maxlen = MAXLEN)
train_set = np.vectorize(chr)(train_set +32)
train_label = np.vectorize(chr)(train_label + 32)
print (train_set[:10])
print(train_label[:10])

#%%
print('Vectorization...')
X = np.zeros((len(train_label), MAXLEN, len(chars)), dtype=np.bool)
y = np.zeros((len(train_label), MAXLEN, len(chars)), dtype=np.bool)
for i, sentence in enumerate(train_set):
    X[i] = ctable.encode(sentence, maxlen=MAXLEN)
for i, sentence in enumerate(train_label):
    y[i] = ctable.encode(sentence, maxlen=MAXLEN)
    
#%%
# Explicitly set apart 10% for validation data that we never train over
split_at = int(len(X) - len(X) / 10)
(X_train, X_val) = (X[:split_at], X[split_at:])
(y_train, y_val) = (y[:split_at], y[split_at:])

print(X_train.shape)
print(y_train.shape)

#%%

print('Build model...')
model = Sequential()
# "Encode" the input sequence using an RNN, producing an output of HIDDEN_SIZE
# note: in a situation where your input sequences have a variable length,
# use input_shape=(None, nb_feature).
model.add(RNN(HIDDEN_SIZE, input_shape=(MAXLEN, len(chars))))
# For the decoder's input, we repeat the encoded input for each time step
model.add(RepeatVector(MAXLEN))
# The decoder RNN could be multiple layers stacked or a single layer
for _ in range(LAYERS):
    model.add(RNN(HIDDEN_SIZE, return_sequences=True))

# For each of step of the output sequence, decide which character should be chosen
model.add(TimeDistributed(Dense(len(chars))))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#%%
class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'

for iteration in range(1, 2):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X_train, y_train, batch_size=BATCH_SIZE, nb_epoch=20,
              validation_data=(X_val, y_val))
    ###
    
    # Select 10 samples from the validation set at random so we can visualize errors
    for i in range(100):
        ind = np.random.randint(0, len(X_val))
        rowX, rowy = X_val[np.array([ind])], y_val[np.array([ind])]
        preds = model.predict_classes(rowX, verbose=0)
        q = ctable.decode(rowX[0])
        correct = ctable.decode(rowy[0])
        guess = ctable.decode(preds[0], calc_argmax=False)
        print('Q', q[::-1] if INVERT else q)
        print('T', correct)
        print(colors.ok + '☑' + colors.close if correct == guess else colors.fail + '☒' + colors.close, guess)
        print('---')

