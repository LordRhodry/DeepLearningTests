# -*- coding: utf-8 -*-
"""
Created on Wed May 24 17:02:39 2017

@author: Julien
"""

# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle

from setupdata import ready_data
ready_data();
  
batch_size = 128
reg_val = 0.0003
dropout_val = 0.5

graph = tf.Graph()
with graph.as_default():

  # Input data. For the training data, we use a placeholder that will be fed
  # at run time with a training minibatch.
  tf_train_dataset = tf.placeholder(tf.float32,shape=(batch_size, image_size * image_size))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)
  
  # Variables.
  weights1 = tf.Variable(tf.truncated_normal([image_size * image_size, 1024],stddev=0.03))
  biases1 = tf.Variable(tf.zeros([1024]))
  weights2 = tf.Variable(tf.truncated_normal([1024, 512],stddev=0.02))
  biases2 = tf.Variable(tf.zeros([512]))
  weights3 = tf.Variable(tf.truncated_normal([512, 128],stddev=0.03))
  biases3 = tf.Variable(tf.zeros([128]))
  weights4 = tf.Variable(tf.truncated_normal([128, num_labels],stddev=0.04))
  biases4 = tf.Variable(tf.zeros([num_labels]))
  
  #define computation:
  def model(data , keep):
      hidden = tf.nn.relu(tf.matmul(data, weights1) + biases1)
      hidden_kept = tf.nn.dropout(hidden , keep)
      hidden = tf.nn.relu(tf.matmul(hidden_kept, weights2) + biases2)
      hidden_kept = tf.nn.dropout(hidden , keep)
      hidden = tf.nn.relu(tf.matmul(hidden_kept, weights3) + biases3)
      hidden_kept = tf.nn.dropout(hidden , keep)
      return tf.matmul(hidden_kept, weights4) + biases4
  
  
  # Training computation.
  logits = model (tf_train_dataset , dropout_val)
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits)  + reg_val / 2 *( tf.nn.l2_loss(weights1)
+   tf.nn.l2_loss(weights2) +   tf.nn.l2_loss(weights3) +   tf.nn.l2_loss(weights4)))
  # Optimizer.
  global_step = tf.Variable(0)  # count the number of steps taken.
  learning_rate = tf.train.exponential_decay(0.5, global_step, 5000, 0.96, staircase=True)
  optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
  
  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(model(tf_valid_dataset,1.0)) 
  test_prediction = tf.nn.softmax(model(tf_test_dataset, 1.0))
  
num_steps = 10001

with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print("Initialized")
  for step in range(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    # Generate a minibatch.
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 500 == 0):
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(
        valid_prediction.eval(), valid_labels))
  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))