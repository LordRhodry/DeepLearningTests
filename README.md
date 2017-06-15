# DeepLearningTests
A directory to store my various experimentations with deep learning

All experimentations were done on my personal computer with the following specs :

CPU : Intel i3-6100 @3.7GHz
RAM : 16GB
GPU : Nvidia GTX 950 with 2GB of VRAM
OS : Win 7 SP1

Using:
Anaconda with a Python 3.5 environment running (include usual data science package : scipy , numpy , matplotlib ...):
Tensorflow-Gpu 1.1.0 (with the CUDA 8.0 and CuDNN 5.1)
Keras 2.0.4

testKeras_MNIST is just a simple test of Keras using the included MNIST dataset

In the NotMNIST folder are various tests in the following order : 
1 - preparedata : download and prepare the data
2 - setupdata : some setup 
3 - firstlearning : simple 1 layer fully connected using gradient descent
4 - firstSGD : simple 1 layer fully connected using batches and SGD
5 - multilayer_l2_dropout_nn : multiple layer, fully connected with some tests on regularization ( the current version of teh file is with L2)
6 - convnn - 2 different models using convolution layer and max pooling
7 - notmnist_keras : Keras implementation of the convolution and maxpool version
8 - testinception : using Keras, a version of inception_v3 cut after the first incepton module for computation cost purpose ( achieved 97.3 on the test set of the NotMNIST) 
9 - conv-RNN-seqofimages : using sequences of 1-5 images from the NotMNIST data , this uses a few layer of convolution and max pool followed by a LSTM to predict a sequence of characters.

In the  RNN folder:
- frankensteining is training a character by character RNN on the novel Frankenstein by Mary Shelley ... results were getting stuck on a repetition : to avoid taht I sued a temperature sample function found in Francois choolet github examples : https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py
-  RNN-bigrams is an experimentation to try to improve prediction by feeding a bigram rather than a single character. the current implementation is cheating to avoid getting stuck on weird bigrams like "  " which were creating bugs.

In the word2vec folder:
- CBOW is a bag od word implementation of word2vec
- skipgram is a skip-gram implementation  
In both of those very little (if any) change have been made from the found examples.

In the seq2seq folder:
- reverse_seq is a silly toy example where data is a sequence of character which label is the same sequence in the reversed order: (the => eht , quick => kciuq ...)
- translation is using the europarl data set to tarin a French to English translation. with 20 epochs result are so far very unsatisfying => some idea not yet implemented to improve this would be to do some pretreatement of the data like removing capital letters and punctuation. The model also hit the limits of the memory on my GPU and I had to limit the vocabulary size and the number and size of the RNN in both the encoder and the decoder. Anotehr idea but that might equire even more data would be to use bucketing to optimize the training depending on the length of sentences. the current version is heavily based on the code from https://chunml.github.io/ChunML.github.io/project/Sequence-To-Sequence/
