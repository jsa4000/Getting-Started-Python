import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt

# TSNE - To see vector images classifications

# Interesting Video Youtube
#  https://www.youtube.com/watch?v=VINCQghQRuM   General Sequence Learning using Recurrent Neural Networks 
#  https://www.youtube.com/watch?v=56TYLaQN4N8   Deep Learning Lecture 12: Recurrent Neural Nets and LSTMs 
#  https://www.youtube.com/watch?v=E92jDCmJNek   Alex Rubinsteyn: Python Libraries for Deep Learning with Sequences 
#  https://www.youtube.com/watch?v=iX5V1WpxxkY   CS231n Lecture 10 - Recurrent Neural Networks, Image Captioning, LSTM 

# https://www.youtube.com/watch?v=_EviCgtzG7E   PyData Paris 2016 - Automatic Machine Learning using Python & scikit-learn 



# http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/
# http://karpathy.github.io/2015/05/21/rnn-effectiveness/
# http://colah.github.io/posts/2015-08-Understanding-LSTMs/

# Here we are going to implement a Recurrent Neural Network.

# Recurrent Neural Networks are more sophistifacted than standard neural networks, where each unit could have different states.
# This is really handly for outputs where each output really depend on the previous one, for that reason is very important
# to store the previous states in memory. To sum up, the RNN are used for sequences of inputs like: text recognition, speech, etc..
#
# These RNN are very expensive to compute due the number of weighs (U,V and W at least) that are needed to train the network. 
# These networks are composed by inputs, hidden layers and output layers. However all of them depend on the number of inputs 
# that represent the possible outputs and possible states. 
#   
#
# In this kind of networks the units will be unfoldded depending on the number of inputs for each sentence. This unfold has 
# similarities with the layers in NN. The units will be connected together and with theirselves (unfolded) but using the
# same paramters. In this case we will have H x H parameteris where H is the number of hidden units.
#    Example.   Units = 5
#              W0    W1    W2    W3    W4
#         W0   0    0.1    0.7   0.4    0.5
#         W1   0    0.2    0.6   1.2    0.3
#         W2   1    0.1    0.7   0.4    0.5
#         W2   0    0.4    1.5   0.2    0.2
#         W3  0.3   0.1    0.7   0.4    0.5

# See following ebook to see examples of the shape Theano\ESNTutorialRev.pdf

# Eventually, previous output and state are transfered. In order to compute the new state in t and the output the inputs_t,
# output_t-1 and the previous state_t-1 will be considered.

# In orther to compute the hidden states (using the previous one t-1 and the input_t) some functions like  tanhor RuLu are
# used. However for the output a softmax fuction is used because a vector with probabilities acorss a vocabulay previous defined.
# Mostly because classification pourposes and because the cross-entropy cost function that will be used for the backprogation.

# For each index of each sequence, the states of the neurons will change so it's necessary to store all of them. Because of 
# this the amount of states and weights for the training is very heavy. The state of an unit it's also called the Memory cell.
#   Note that for the first state, the previous one isn't exit yet. So, initial states with initilized values must be added
#   to do the computation correctly. This is done with zeros.

# RNN shares the same parameters (U,V and W) across all steps. This is because the task is performed is the same all the time.
# As previously said not always the RNN is going to need outputs and inputs.

# For the definition of a basic RNN each unit will have three weights that will be trained:
# - Inputs to Units. U
# - From one state to another W
# - Unit to Outputs. V

# if the RNN has more that 1 layer then U1, W1, V1, ... UN, WN and V1

# The States and Outputs for each unit will be computed in base of those weights and Inputs.
# For each input, the neural network will generate new outputs that will be feeding into the newer states. 
# A COMPLETE sequence is for example a sentence of 5 words, this mean the RNN would be enrolled into a 5 layers. 

#For example if we have a recurrent neural network like the following:
# inputs (xt) = 8000
# hidden_layer (st) = 100 (only one layer, but could be connected more of it would be neccesary)
# outputs (ot) = 8000 (dependd on the topoloy of the recurrent network and inputs, many-to-one, many-to-many, one_to-many, etc..)

# The wight will have the following size
#  U: 8000x100
#  W: 100x100
#  V: 100x8000

# Basic Equations
# st = tanh (Uxt + Wst-1) -> hidden t state f the unit -> ((8000x100)T dot (8000,1)) + (100,100) dot  (100, 1)
# ot = softmax (Vst) -> output t of the unit

# TEXT Recognition

## Following there is a training data x and the output y.

#  First we create the dictionary for the possible inputs (8000)
#  Using only hot-vector means if A, B, C, D -> 1 0 0 0, 0 1 0 0, 0 0 1 0, 0 0 0 1. Array (4,)
# In this case we need a vocabulary with words so: Hello, My name is Javier. This will turn into
#               Hello, My, Name, is, Javier
#                 0     1    2    3    4          # (5,)
#
#              1 0 0 0 0   -> Hello
#              0 1 0 0 0   -> My
#              0 0 1 0 0   -> Name
#              0 0 0 1 0   -> is
#              0 0 0 0 1   -> Javier
#
# 8000 means we have a Matrix with 8000 possible words. -> Vocabulary size
# How ever the sequence we receive is (0, 1, 2, 3, 4)  or [45, 156, 34, 723, 56, 456, 324] etc...


# Following is a real example
#x:
#SENTENCE_START what are n't you understanding about this ? !
#[0, 51, 27, 16, 10, 856, 53, 25, 34, 69]
 
#y:
#what are n't you understanding about this ? ! SENTENCE_END
#[51, 27, 16, 10, 856, 53, 25, 34, 69, 1]

# So that's mean we are going to feed the word with index [input + 1 ] xi+1 = oi . This means the next output we want is the next
# that it's coming in the sentence. e. Hola que tal estás?. 
#   input x0 = Hola ; Output o0 = que
#   input x1 = que ; Output o1 = tal
#   input x0 = tal ; Output o0 = estás  ... so on


