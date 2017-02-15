import theano
import theano.typed_list
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt


######################
### THEANO   
#######################


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
       return theano.shared(floatX(np.zeros(shape)))
    elif (mode == "U"):
        return theano.shared(floatX(np.random.uniform(-np.sqrt(1./shape[0]), np.sqrt(1./shape[0]), shape)))
    else:
        return theano.shared(floatX(np.random.rand(*shape) * 0.1))

# For Recurrent Neural Network, following components are required.
x = T.matrix() # Inputs with the sequences NixMi, where m are the index of the elements of the sequence x1,x2,x3, etc...
t = T.matrix() # Outputs with the sequences or outputs, NoxMo, where m are the index of the elements of the sequence o1,o2,o3, etc..

n = theano.tensor.lvector()

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

shape = (4,3,2,4)  
functions = ((T.tanh,T.tanh), (T.tanh,T.nnet.softmax))

layerCount = len(shape)

# Matrix initialization. This is very important to train the network and optimize the training
# There are 3 params to compute for each hidden layer W, U and V
U = []
W = []
V = []
# compute only the hidden layers to create the params
for index in range (1, layerCount-1):
    # creates the input width parameters between this layer and the previous one 
    U.append(init_weigth((shape[index], shape[index-1]),"U"))   
    # creates the hidden width parameters for the current layer
    W.append(init_weigth((shape[index], shape[index]),"U"))
    # creates the output parameters between this layer and the next one 
    V.append(init_weigth((shape[index+1], shape[index]),"U"))

#For layer I mean hidder layer
def layersstep(x):
    """
         Function to implement with Step using theano for each hidden layer configured
         For that reason we need to look into the RNN algorithm and get how it works. This will depend on how LSTM, GRU, etc.. works.

         Parameters:
            layer: number of the hidden layer being computed. Thiss is to take the proper function, W, U and V
            x: previous input from another layer o from the inputs sequence
            ps: previous state of this layer
            U, W, V: matrixes with the params to update. An index with the later is neccesary in order to get the params for each layer
    """
    #  st = tanh (Uxt + Wst-1)
    st = functions[layer.get][0](T.dot(U[layer],x) + T.dot(W[layer],ps[layer]))
    # ot = softmax (Vst)
    ot = T.transpose(functions[layer][1] (T.dot(U[layer],st)))
    #Return current output and current state
    return ot, st


# Define the function. Shared variables are not going to be passed as parameters.
y = layerstep(n, x)

# Compitel the function
func = theano.function([n,x], y) 

# Lets Define the first input
input = np.asarray([0, 0, 1, 0])
ot, st = func([0],input)

print (func.eval())







######################
### NUMPY   
#######################

#DictionarySize = 1000  

#def init_nweigth(shape,mode = "R"):
#    """ Returns a np array with the shae defined
#        Parameters:
#            mode: "Z": zeros theano shared array with the shape
#                  "R": random theano shared array with the shape
#                  "U": uniform theano shared array with the shape and some parameters ising the first index to random generation
#    """
#    if (mode == "Z"):
#       return np.zeros(shape)
#    elif (mode == "U"):
#       return np.random.uniform(-np.sqrt(1./shape[0]), np.sqrt(1./shape[0]), shape)
#    else:
#        return np.random.rand(*shape) * 0.1
#def softmax(x):
#    return np.exp(x) / np.sum(np.exp(x))

##Define the shape of the RNN, with the possible inputs/outputs and the recurrent units.
#shape = (4,3,2,4)  
## For each hidden units define the functions that will be applied for the inputs and the outputs
#nfunctions = ((np.tanh,np.tanh), (np.tanh,softmax))
#layerCount = len(shape)

## Matrix initialization. This is very important to train the network and optimize the training
## There are 3 params to compute for each hidden layer W, U and V
#nU = []
#nW = []
#nV = []
## compute only the hidden layers to create the params
#for index in range (1, layerCount-1):
#    # creates the input width parameters between this layer and the previous one 
#    nU.append(init_nweigth((shape[index], shape[index-1]),"U"))   
#    # creates the hidden width parameters for the current layer
#    nW.append(init_nweigth((shape[index], shape[index]),"U"))
#    # creates the output parameters between this layer and the next one 
#    nV.append(init_nweigth((shape[index+1], shape[index]),"U"))

## Once we have all the paramters into place, next is to define each step, based on the scan operator in thano
## For that reason we need to look into the RNN algrithm and get how it works. This will depend on how LSTM, GRU, etc.. works.
#def nstep( x, ps, U, W, V):
#    # For each equation y need to do the folloing equations
#    # st = tanh (Uxt + Wst-1) -> hidden t state f the unit -> ((8000x100)T dot (8000,1)) + (100,100) dot  (100, 1)
#    # ot = softmax (Vst) -> output t of the unit

#    cs = []
#    # We have to iterate between the layer, theano does'n allow this kind of operations sometimes.
#    for layer in range(layerCount - 2):
#        #Take the functions and state for each equation
#        #  st = tanh (Uxt + Wst-1)
#        st = nfunctions[layer][0](np.dot(U[layer],x) + np.dot(W[layer],ps[layer]))
#        # ot = softmax (Vst)
#        x = nfunctions[layer][1] (np.dot(V[layer],st)).T
#        # Append the current state
#        cs.append(st)

#    return x, cs

## Start declaring the variables for numerical computing

## Declare initial states for the units inside the layers

#ps = []
## compute only the hidden layers to create the params
#for index in range (1, layerCount-1):
#    # creates the hidden width parameters for the current layer
#    ps.append(init_nweigth((shape[index], 1),"Z"))
## Lets Define the first input
#x = [0, 0, 1, 0]


#y, fs = nstep(x, ps, nU, nW, nV)

# Returns a 2x4 matrix


