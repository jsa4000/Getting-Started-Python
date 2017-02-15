import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt

# Start declaring the symbolic variables to be compiled by theano

def floatX (x):
    return np.asarray(x,theano.config.floatX)

def init_weigth(shape,mode = "R"):
    """ Returns a np array with the shae defined
        Parameters:
            mode: "Z": zeros theano shared array with the shape
                  "R": random theano shared array with the shape
                  "U": uniform theano shared array with the shape and some parameters ising the first index to random generation
    """
    if (mode == "Z"):
       return theano.shared(floatX(np.zeros_like(shape)))
    elif (mode == "U"):
        #return theano.shared(floatX(np.random.uniform(-np.sqrt(1./shape[0]), np.sqrt(1./shape[0]), shape)))
        return np.random.uniform(-np.sqrt(1./shape[0]), np.sqrt(1./shape[0]), shape)
    else:
        return theano.shared(floatX(np.random.rand(*shape) + 0.1))

# For Recurrent Neural Network, following components are required.
x = T.matrix() # Inputs with the sequences NixMi, where m are the index of the elements of the sequence x1,x2,x3, etc...
y = T.matrix() # Outputs with the sequences or outputs, NoxMo, where m are the index of the elements of the sequence o1,o2,o3, etc..

# For the Params we must initialize the shape of the entire Network, since this params are declared as shared variables in theano

# First we create the dictionary for the possible inputs/outputs decoded as hot-vectors (1000),
#      - where hot vector means if A, B, C, D -> 1 0 0 0, 0 1 0 0, 0 0 1 0, 0 0 0 1. Array (4,)
# DictionarySize = 1000 means we have a Matrix with 1000 possible items from 0 to 999. -> Dictionary size
# How ever the sequence we receive/generate is [0, 1, 2, 3, 4]  or [45, 156, 34, 723, 56, 456, 324] etc...
DictionarySize = 1000  

#Define the shape of the RNN, with the possible inputs/outputs and the recurrent units.
shape = (DictionarySize,100,200, DictionarySize)  
# For each hidden units define the functions that will be applied for the inputs and the outputs
functions = ((T.tanh,T.nnet.softmax))  #  ((T.tanh,T.nnet.relu), (T.tanh,T.nnet.softmax))
layerCount = len(shape)

# Matrix initialization. This is very important to train the network and optimize the training
# There are 3 params to compute for each hidden layer W, U and V
W = []
U = []
V = []
# compute only the hidden layers to create the params
for index in range (1, layerCount-1):
    # creates the input width parameters between this layer and the previous one 
    U.append(init_weigth((shape[index-1], shape[index]),"U"))   
    # creates the hidden width parameters for the current layer
    W.append(init_weigth((shape[index], shape[index]),"U"))
    # creates the output parameters between this layer and the next one 
    V.append(init_weigth((shape[index], shape[index + 1]),"U"))

# Once we have all the paramters into place, next is to define each step, based on the scan operator in thano
# For that reason we need to look into the RNN algrithm and get how it works. This will depend on how LSTM, GRU, etc.. works.
def step( x, ps, weights):
    # For each equation y need to do the folloing equations
    # st = tanh (Uxt + Wst-1) -> hidden t state f the unit -> ((8000x100)T dot (8000,1)) + (100,100) dot  (100, 1)
    # ot = softmax (Vst) -> output t of the unit

    cs = theano.typed_list.TypedListType(T.fscalar)()
    # We have to iterate between the layer, theano does'n allow this kind of operations sometimes.
    for layer in range(layerCount - 2):
        #Take the functions and state for each equation
        #  st = tanh (Uxt + Wst-1)
        st = functions[layer][0](T.dot(U[layer],x) + T.dot(W[layer],theano.typed_list.basic.getitem(ps,layer)))
        theano.typed_list.append(cs,st)
        # ot = softmax (Vst)
        x = functions[layer][1] (T.dot(U[layer],st))

    return x, cs

# Start declaring the variables for numerical computing



