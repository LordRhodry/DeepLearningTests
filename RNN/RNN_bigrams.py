# -*- coding: utf-8 -*-
"""
Created on Mon May 29 17:47:27 2017

@author: Julien
"""

# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import collections
import os
import numpy as np
import random
import string
import tensorflow as tf
import zipfile
from six.moves import range
from six.moves.urllib.request import urlretrieve

#%%
url = 'http://mattmahoney.net/dc/'

def maybe_download(filename, expected_bytes):
  """Download a file if not present, and make sure it's the right size."""
  if not os.path.exists(filename):
    filename, _ = urlretrieve(url + filename, filename)
  statinfo = os.stat(filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified %s' % filename)
  else:
    print(statinfo.st_size)
    raise Exception(
      'Failed to verify ' + filename + '. Can you get to it with a browser?')
  return filename

filename = maybe_download('text8.zip', 31344016)

#%%
def read_data(filename):
  with zipfile.ZipFile(filename) as f:
    name = f.namelist()[0]
    data = tf.compat.as_str(f.read(name))
  return data
  
text = read_data(filename)
print('Data size %d' % len(text))

#%%
valid_size = 1000
valid_text = text[:valid_size]
train_text = text[valid_size:]
train_size = len(train_text)
print(train_size, train_text[:64])
print(valid_size, valid_text[:64])

#%%

vocabulary_size = len(string.ascii_lowercase) + 1 # [a-z] + ' '
first_letter = ord(string.ascii_lowercase[0])

def char2id(char):
  if char in string.ascii_lowercase:
    return ord(char) - first_letter + 1
  elif char == ' ':
    return 0
  else:
    print('Unexpected character: %s' % char)
    return 0
  
def id2char(dictid):
  if dictid > 0:
    return chr(dictid + first_letter - 1)
  else:
    return ' '

print(char2id('a'), char2id('z'), char2id(' '), char2id('ï'))
print(id2char(1), id2char(26), id2char(0))

#%%
bigrams = [''.join(x) for x in zip(train_text[:-1] , train_text[1:])]
print (bigrams[:64])

#%%
vocabulary_size = 27*27

def build_dataset(bigrams):
  
  count = (collections.Counter(bigrams).most_common(vocabulary_size))
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys())) 
  return dictionary, reverse_dictionary

dictionary, reverse_dictionary = build_dataset(bigrams)

del bigrams


#%%
batch_size=64
num_unrollings=10
vocabulary_size = len(string.ascii_lowercase) + 1 # [a-z] + ' '

class BatchGenerator(object):
  def __init__(self, text, batch_size, num_unrollings):
    self._text = text
    self._text_size = len(text)
    self._batch_size = batch_size
    self._num_unrollings = num_unrollings
    segment = self._text_size // batch_size
    self._cursor = [ offset * segment for offset in range(batch_size)]
    self._last_batch , self._last_label = self._next_batch()
  
  def _next_batch(self):
    """Generate a single batch from the current cursor position in the data."""
    batch = list()
    del batch[:]
    label = np.zeros(shape=(self._batch_size, vocabulary_size), dtype=np.float)
    for b in range(self._batch_size):
      batch.append(dictionary[self._text[self._cursor[b]] + self._text[(self._cursor[b]+1) % self._text_size]])
      label[b, char2id(self._text[(self._cursor[b]+2)  % self._text_size])] = 1.0
      self._cursor[b] = (self._cursor[b] + 1) % self._text_size
    return batch , label
  
  def next(self):
    """Generate the next array of batches from the data. The array consists of
    the last batch of the previous array, followed by num_unrollings new ones.
    """
    batches = [self._last_batch]
    labels = [self._last_label]
    for step in range(self._num_unrollings):
      temp_batch , temp_label = self._next_batch()
      batches.append(temp_batch)
      labels.append(temp_label)
    self._last_batch = batches[-1]
    self._last_label = labels[-1]
    return batches ,labels

def characters(probabilities):
  """Turn a 1-hot encoding or a probability distribution over the possible
  characters back into its (most likely) character representation."""
  return [id2char(c) for c in np.argmax(probabilities, 1)]

def batches2string(batches):
  """Convert a sequence of batches back into their (most likely) string
  representation."""
  s = [''] * batches[0].shape[0]
  for b in batches:
    s = [''.join(x) for x in zip(s, characters(b))]
  return s

train_batches = BatchGenerator(train_text, batch_size, num_unrollings)
valid_batches = BatchGenerator(valid_text, 1, 1)

batch, label = train_batches.next()
print('labels: ')
print(len(label[0]))
print( batches2string(label))
print('batch: ')
print(len(batch[0]))
print( batch)
batch, label = train_batches.next()
print('labels: ')
print( batches2string(label))
print('batch: ')
print( batch)
batch, label = valid_batches.next()
print(batches2string(label))
batch, label = valid_batches.next()
print(batches2string(label))

#%%
def logprob(predictions, labels):
  """Log-probability of the true labels in a predicted batch."""
  predictions[predictions < 1e-10] = 1e-10
  return np.sum(np.multiply(labels, -np.log(predictions))) / labels.shape[0]

def sample_distribution(distribution):
  """Sample one element from a distribution assumed to be an array of normalized
  probabilities.
  """
  r = random.uniform(0, 1)
  s = 0
  for i in range(len(distribution)):
    s += distribution[i]
    if s >= r:
      return i
  return len(distribution) - 1

def sample(prediction):
  """Turn a (column) prediction into 1-hot encoded samples."""
  p = np.zeros(shape=[1, vocabulary_size], dtype=np.float)
  p[0, sample_distribution(prediction[0])] = 1.0
  return p

def random_distribution():
  """Generate a random column of probabilities."""
  b = np.random.randint(0,160)
  return b

#%% Simple LSTM model

num_nodes = 64
dropout_rate = 1

graph = tf.Graph()
with graph.as_default():
  
  # Parameters:
  # Input gate: input, previous output, and bias.
  #ix = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1))
  #im = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
  ib = tf.Variable(tf.zeros([1, num_nodes]))
  # Forget gate: input, previous output, and bias.
  #fx = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1))
  #fm = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
  fb = tf.Variable(tf.zeros([1, num_nodes]))
  # Memory cell: input, state and bias.                             
  #cx = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1))
  #cm = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
  cb = tf.Variable(tf.zeros([1, num_nodes]))
  # Output gate: input, previous output, and bias.
  #ox = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1))
  #om = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
  ob = tf.Variable(tf.zeros([1, num_nodes]))
  # Variables saving state across unrollings.
  saved_output = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
  saved_state = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
  # Classifier weights and biases.
  w = tf.Variable(tf.truncated_normal([num_nodes, vocabulary_size], -0.1, 0.1))
  b = tf.Variable(tf.zeros([vocabulary_size]))
  matx = tf.Variable(tf.truncated_normal([vocabulary_size, 4*num_nodes], -0.1, 0.1))
  matm = tf.Variable(tf.truncated_normal([num_nodes, 4*num_nodes], -0.1, 0.1))
  
  embeddings = tf.Variable(tf.random_uniform([vocabulary_size **2 , vocabulary_size] , -1.0, 1.0))
  
  # Definition of the cell computation.
  def lstm_cell(i, o, state , keep_prob):
    """Create a LSTM cell. See e.g.: http://arxiv.org/pdf/1402.1128v1.pdf
    Note that in this formulation, we omit the various connections between the
    previous state and the gates."""
    inputmat = tf.nn.embedding_lookup(embeddings, i)
    inputmat = tf.matmul(inputmat,matx)
    outputmat = tf.matmul(o,matm)
    input_gate =tf.nn.dropout( tf.sigmoid(inputmat[:,:num_nodes] + outputmat[:,:num_nodes] + ib) , keep_prob)
    forget_gate = tf.sigmoid(inputmat[:,num_nodes:2*num_nodes] + outputmat[:,num_nodes:2*num_nodes] + fb)
    update = inputmat[:,2*num_nodes:3*num_nodes] + outputmat[:,2*num_nodes:3*num_nodes] + cb
    state = forget_gate * state + input_gate * tf.tanh(update)
    output_gate = tf.nn.dropout(tf.sigmoid(inputmat[:,3*num_nodes:] + outputmat[:,3*num_nodes:] + ob),keep_prob)
    return output_gate * tf.tanh(state), state

  # Input data.
  train_data = list()
  train_label_data = list()
  for _ in range(num_unrollings ):
    train_data.append(
      tf.placeholder(tf.int32, shape=[batch_size,]))
    train_label_data.append(
      tf.placeholder(tf.float32, shape=[batch_size,vocabulary_size]))
  train_inputs = train_data
  train_labels = train_label_data  # labels are inputs shifted by one time step.

  # Unrolled LSTM loop.
  outputs = list()
  output = saved_output
  state = saved_state
  for i in train_inputs:
    output, state = lstm_cell(i, output, state, dropout_rate)
    outputs.append(output)

  # State saving across unrollings.
  with tf.control_dependencies([saved_output.assign(output),
                                saved_state.assign(state)]):
    # Classifier.
    logits = tf.nn.xw_plus_b(tf.concat(outputs, 0), w, b)
    loss = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(
        labels=tf.concat(train_labels, 0), logits=logits))

  # Optimizer.
  global_step = tf.Variable(0)
  learning_rate = tf.train.exponential_decay(
    10.0, global_step, 3000, 0.3, staircase=True)
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  gradients, v = zip(*optimizer.compute_gradients(loss))
  gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
  optimizer = optimizer.apply_gradients(
    zip(gradients, v), global_step=global_step)

  # Predictions.
  train_prediction = tf.nn.softmax(logits)
  
  # Sampling and validation eval: batch 1, no unrolling.
  sample_input = tf.placeholder(tf.int32, shape=[1, ])
  saved_sample_output = tf.Variable(tf.zeros([1, num_nodes]))
  saved_sample_state = tf.Variable(tf.zeros([1, num_nodes]))
  reset_sample_state = tf.group(
    saved_sample_output.assign(tf.zeros([1, num_nodes])),
    saved_sample_state.assign(tf.zeros([1, num_nodes])))
  sample_output, sample_state = lstm_cell(
    sample_input, saved_sample_output, saved_sample_state, 1.0)
  with tf.control_dependencies([saved_sample_output.assign(sample_output),
                                saved_sample_state.assign(sample_state)]):
    sample_prediction = tf.nn.softmax(tf.nn.xw_plus_b(sample_output, w, b))
    
#%% 

num_steps = 10001
summary_frequency = 100

with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print('Initialized')
  mean_loss = 0
  for step in range(num_steps):
    batches , labels_temp = train_batches.next()
    feed_dict = dict()
    for i in range(num_unrollings):
      feed_dict[train_data[i]] = batches[i]
      feed_dict[train_label_data[i]] = labels_temp[i]
    _, l, predictions, lr = session.run(
      [optimizer, loss, train_prediction, learning_rate], feed_dict=feed_dict)
    mean_loss += l
    if step % summary_frequency == 0:
      if step > 0:
        mean_loss = mean_loss / summary_frequency
      # The mean loss is an estimate of the loss over the last few batches.
      print(
        'Average loss at step %d: %f learning rate: %f' % (step, mean_loss, lr))
      mean_loss = 0
      labels = np.concatenate(list(labels_temp[:-1]))
      print('Minibatch perplexity: %.2f' % float(
        np.exp(logprob(predictions, labels))))
      if step % (summary_frequency * 10) == 0:
        # Generate some samples.
        print('=' * 80)
        for _ in range(5):
          feed = random_distribution()
          sentence = reverse_dictionary[feed]
          reset_sample_state.run()
          for _ in range(79):
            prediction = sample_prediction.eval({sample_input:np.array( [feed])})
            feed = sample(prediction)
            sentence += characters(feed)[0]
            if sentence[-2:] == '  ':
                feed = random_distribution()
            else:
                feed = dictionary[sentence[-2:]]
          print(sentence)
        print('=' * 80)
      # Measure validation set perplexity.
      reset_sample_state.run()
      valid_logprob = 0
      for _ in range(valid_size):
        a , b = valid_batches.next()
        predictions = sample_prediction.eval({sample_input: a[0]})
        valid_logprob = valid_logprob + logprob(predictions, b[0])
      print('Validation set perplexity: %.2f' % float(np.exp(
        valid_logprob / valid_size)))

