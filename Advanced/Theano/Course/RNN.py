import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt

# Declaring some functions to create the params for the weights to update in the NN
def floatX (x):
    return np.asarray(x,theano.config.floatX)

def init_weigth(shape,mode = "R"):
    """ Returns a np array with the shae defined
        Parameters:
            mode: "Z": zeros theano shared array with the shape
                  "R": random theano shared array with the shape
                  "O": ones theano shared array with the shape
                  "U": uniform theano shared array with the shape
    """
    if (mode == "Z"):
       return theano.shared(floatX(np.zeros(shape)))
    elif (mode == "O"):
       return theano.shared(floatX(np.ones(shape)))
    elif (mode == "U"):
        return theano.shared(floatX(np.random.uniform(-np.sqrt(1./shape[1]), np.sqrt(1./shape[1]), shape)))
    else:
        return theano.shared(floatX(np.random.rand(*shape) * 0.1))

# Start declaring the symbolic variables to be compiled by theano

# For the Params we must initialize the shape of the entire Network, since this params are declared as shared variables in theano

# First we create the dictionary for the possible inputs/outputs decoded as hot-vectors (1000),
#      - where hot vector means if A, B, C, D -> 1 0 0 0, 0 1 0 0, 0 0 1 0, 0 0 0 1. Array (4,)
# DictionarySize = 1000 means we have a Matrix with 1000 possible items from 0 to 999. -> Dictionary size
# How ever the sequence we receive/generate is [0, 1, 2, 3, 4]  or [45, 156, 34, 723, 56, 456, 324] etc...
DictionarySize = 800  

# Back propagation through time truncate option for optimize RNN
bptt_truncate = 4

#Define the shape of the RNN, with the possible inputs/outputs and the recurrent units.
shape = [DictionarySize,100, DictionarySize]
# For each hidden units define the functions that will be applied for the inputs and the outputs
functions = [[T.tanh,T.nnet.softmax]] #  ((T.tanh,T.nnet.relu), (T.tanh,T.nnet.softmax))
layerCount = len(shape)

# Matrix initialization. This is very important to train the network and optimize the training
# There are 3 params to compute for each hidden layer W, U and V
W = []
U = []
V = []
# compute only the hidden layers to create the params
for index in range (1, layerCount-1):
    # creates the input width parameters between this layer and the previous one 
    U.append(init_weigth((shape[index], shape[index+1]),"U"))   
    # creates the hidden width parameters for the current layer
    W.append(init_weigth((shape[index], shape[index]),"U"))
    # creates the output parameters between this layer and the next one 
    V.append(init_weigth((shape[index+1], shape[index]),"U"))

# Once we have all the paramters into place, next is to define each step, based on the scan operator in thano
# For that reason we need to look into the RNN algrithm and get how it works. This will depend on how LSTM, GRU, etc.. works.
def forward_step( x, *ps):
    """
        Step for fordward propagation using the shape of the NN
        Parameters:
            x: is the input of the sequence that the first layer will take
            ps: are going to be the previous states for each layer
        Outputs:
            result: list with the final output in current NN and current states of each layer
    """
    # For each equation y need to do the folloing equations (Basic RNN)
    # st = tanh (Uxt + Wst-1) -> hidden t state f the unit -> ((8000x100)T dot (8000,1)) + (100,100) dot  (100, 1)
    # ot = softmax (Vst) -> output t of the unit
    states = []
    # We have to iterate between the layers, theano does'n allow this kind of operations sometimes.
    for layer in range(layerCount - 2):
        #Take the functions and state for each equation
        #  st = tanh (Uxt + Wst-1)
        states.append(functions[layer][0](U[layer][:,x] + T.dot(W[layer],ps[layer])))
        # ot = softmax (Vst)
        x = functions[layer][1] (T.dot(V[layer],states[-1]))
    
    #Finally compose the final results for the outpus_info required
    result = []
    result.append(x[0])
    result.extend(states)
    return result

# Initalize previous states for the fordward propagation for each layer (TEnsor variables not tensor shared variables)
initialStates = [] 
initialStates.append(None)
for index in range (1, layerCount-1):
    initialStates.append(dict(initial=T.zeros(shape[index])))  # Are going to be multipled for the W matrix that its NxN where N is the layer's size

# For Recurrent Neural Network, following components are required.
# integer bacause it ill used for scan and hot-vector
x = T.ivector() # Inputs with the sequences NixMi, where m are the index of the elements of the sequence x1,x2,x3, etc...
y = T.ivector() # Outputs with the sequences or outputs, NoxMo, where m are the index of the elements of the sequence o1,o2,o3, etc..

# Declare the scan function in theano because optimization and differentiation for the gradient descent
output, updates = theano.scan(
            forward_step,
            sequences=x,
            truncate_gradient=bptt_truncate, 
            outputs_info=initialStates)

# Defin the const function
cost = T.sum(T.nnet.categorical_crossentropy(output[0],y))

# Define the updates with the gradients and parameters to update

# For the cost we will use SGD. This is no very optimized like using RPrep, but since we are using bptt we will have better performances.
# Extract th gradients for all gradients defined in the model
dU = T.grad(cost,U)
dW = T.grad(cost,W)
dV = T.grad(cost,V)
# create a list for all the grafients extracted
d = []
d.extend(dU)
d.extend(dW)
d.extend(dV)

# Assign functions    
prediction = T.argmax(output[0], axis=1)

# RNN must be trained sequence by sequence instead give the data and train like other NN
predict = theano.function([x], prediction)
forward_propagation = theano.function([x], output[0])
ce_error = theano.function([x, y], cost)
bptt = theano.function([x, y], d)

# Create the updates for the gradient descent for each param
learning_rate = T.scalar()
# updates = W, w - learning_rate * grafient
updates = []
#For each gradient (deriative) extracted (lists) we have to create the updates for the current RNN
for u, g in zip(U,dU):
    updates.append((u, u - learning_rate * g))
for w, g in zip(W,dW):
    updates.append((w, w - learning_rate * g))
for v, g in zip(V,dV):
    updates.append((v, v - learning_rate * g))

# SGD
sgd_step = theano.function([x,y,learning_rate], [], updates=updates)
    
def calculate_total_loss(X, Y):
    return np.sum([ce_error(x,y) for x,y in zip(X,Y)])
    
def calculate_loss(X, Y):
    # Divide calculate_loss by the number of words
    num_words = np.sum([len(y) for y in Y])
    return calculate_total_loss(X,Y)/float(num_words)   

#TEST the data with the feeding

# Start declaring the variables for numerical computing
#RNN.sgd_step([45, 156, 34, 723, 56, 456, 324], [45, 156, 34, 723, 56, 456, 324], 0.005)



