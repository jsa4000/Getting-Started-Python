import theano
import theano.typed_list
import theano.tensor as T
import numpy as np

#################
##  EXAMPLE 1
#################

# http://www.programcreek.com/python/example/61556/theano.scan

#def floatX (x):
#    return np.asarray(x,theano.config.floatX)

## Define the symbolic variables
#x = T.scalar()
##n = T.iscalar()

#shape = (3,3)
#W = theano.shared(floatX(np.random.rand(*shape)))

#def addsequence(n,x,s,W):
#    return (x+1)*n, (s+1)*n

#steps = 3
#n = np.asarray(range(steps))
#ps = np.asarray([0, 0])
#print (n)

#outputs, _ = theano.scan(addsequence,
#                               sequences = [n,ps],
#                               non_sequences = [W],
#                               n_steps = steps)

#f = theano.function([x], outputs)

#################
##  EXAMPLE 2
#################

#x = theano.tensor.vector()
#u = T.scalar()

#hidden_layers = 2

#def polynomial(x, power, W):
#   return x * (W ** power)

## Generate the components of the polynomial
#outputs, _ = theano.scan(polynomial,
#                                  outputs_info=None,
#                                  sequences=[x, theano.tensor.arange(hidden_layers)],
#                                  non_sequences=u)
## Compile a function
#func = theano.function([x, u], outputs)

## Test
#test = numpy.asarray([1, 0, 2], dtype=numpy.float32)
#value = 3
#print(func(test, value))

#################
##  EXAMPLE 2
#################

## Define the functions. Numply array are needded ?
#Functions = [ [T.tanh, T.tanh ] ,
#              [T.tanh, T.tanh ] ,
#              [T.tanh, T.nnet.softmax ]]     

#nFunctions = [[np.tanh, np.tanh ] ,
#              [np.tanh, np.tanh ], 
#              [np.tanh, np.tanh ]]    

##Create the input
#Inputs = np.array(np.linspace(0,8,num = 9), dtype = float).reshape((3,3))      

##Create a Share array 
#W = np.ones((3,3))
#nLayers = 3
#Layers = np.array(range(nLayers),dtype=float)
#print (Layers)

#def op(x,w,n):
#    return nFunctions[n][0]((x * w)) + Layers[n]

#for i in range(nLayers):
#    print op(Inputs,W,i)


#x = T.matrix()
#l = T.vector()

#W = theano.shared(np.ones((3,3),dtype = theano.config.floatX))

#state = theano.shared(1)
#inc = T.iscalar('inc')

#def op(x,l):
#    index = state.se
#    return Functions[index][0]((x * W)) + l[index], l[index]
   
#y = op(x,l)
#func = theano.function([x, l,inc], y, updates=[(state, state+inc)])

#for i in range(nLayers):
#    print(state.get_value())
#    print func(Inputs,Layers,1)



#################
##  EXAMPLE 3
#################

#import theano
#from theano import tensor as T
#from theano.compile.io import In
#x = T.scalar()
#y = T.scalar('y')
#z = T.scalar('z')
#w = T.scalar('w')

#fn = theano.function(inputs=[x, y, In(z, value=42), ((w, w+x), 0)],
#                     outputs=x + y + z)

#print fn(1, y=2) 



#################
##  EXAMPLE 4
#################

#import theano
#import theano.tensor as T
#import numpy as np

## Define the functions. Numply array are needded ?
#Functions = [ [T.tanh, T.tanh ] ,
#              [T.tanh, T.tanh ] ,
#              [T.tanh, T.nnet.softmax ]]     

#nFunctions = [[np.tanh, np.tanh ] ,
#              [np.tanh, np.tanh ], 
#              [np.tanh, np.tanh ]]    

##Create the input
#Inputs = np.array(np.linspace(0,8,num = 9), dtype = float).reshape((3,3))      

##Create a Share array 
#W = np.ones((3,3))
#nLayers = 3
#Layers = np.array(range(nLayers),dtype=float)
#print (Layers)

#def op(x,w,n):
#    return nFunctions[n][0]((x * w)) + Layers[n]

#for i in range(nLayers):
#    print op(Inputs,W,i)
    
#x = T.matrix()
#l = T.vector()
#s = theano.tensor.lvector()
#W = theano.shared(np.ones((3,3),dtype = theano.config.floatX))
#Indexes = theano.shared(np.array(range(3),dtype = 'int32'))

#p = []
#p.append( theano.shared(np.ones((3,3),dtype = theano.config.floatX)))
#p.append( theano.shared(np.ones((4,3),dtype = theano.config.floatX)))
#p.append( theano.shared(np.ones((3,4),dtype = theano.config.floatX)))

## It's not possible using the index of a list using Tensor, no TensorShares nor TensorVariables
## So Functions where the scope it's outside the function it's difficult to deal with.


#def op(x,l,shape):
#    index = shape.shape[0]
#    index = Indexes[index]
#    #return Functions[T.sqr(index)][0]((x * W)) + l[index]
#    #return Functions[index][0](x * W) + l[index]
#    #return T.tanh(x * W) + l[index], index
#    return T.tanh(x * W) + l[index] + p[index],

#y = op(x,l,s)
#func = theano.function([x,l,s], y)

#for i in range(nLayers):
#    print func(Inputs,Layers,np.array(range(i),dtype = 'int32'))


##################
###  EXAMPLE 4
##################

#import theano
#import theano.tensor as T
#import numpy as np

## Define the functions. Numply array are needded ?
#Functions = [ [T.tanh, T.tanh ] ,
#              [T.tanh, T.tanh ] ,
#              [T.tanh, T.nnet.softmax ]]     


    
#x = T.vector()
#l = T.vector()

#nLayers = 3
##Create the input
#Inputs = np.array(np.linspace(0,8,num = 9), dtype = float).reshape((3,3))  
##Create a Layers array 
#Layers = np.array(range(nLayers),dtype = theano.config.floatX)

#W = []
#W.append( theano.shared(np.ones((3,3),dtype = theano.config.floatX)))
#W.append( theano.shared(np.ones((3,4),dtype = theano.config.floatX)))
#W.append( theano.shared(np.ones((3,5),dtype = theano.config.floatX)))

#p = []
#p.append( theano.shared(np.ones((3,4),dtype = theano.config.floatX)))
#p.append( theano.shared(np.asarray(np.random.randn(*(4,4)),dtype = theano.config.floatX)))
#p.append( theano.shared(np.ones((5,4),dtype = theano.config.floatX)))

## It's not possible using the index of a list using Tensor, no TensorShares nor TensorVariables
## So Functions where the scope it's outside the function it's difficult to deal with.
#def op(x,l):
#    value = []
#    for index in range(nLayers):
#        value.append( T.dot( Functions[index][0]( x.dot(W[index]) ),p[index]) + l[index] )
#    return value

#y = op (x,l)
#func = theano.function([x,l], y)

#print func(Inputs[0],Layers)
##print func(Inputs[0][1],Layers)
##print func(Inputs[0][1],Layers)

#def op(x,l,p):
#    value = []
#    for index in range(nLayers):
#        #value.append( T.dot( Functions[index][0]( x * W[index] ),p[index]) + l[index] )
#        value.append( Functions[index][0]( x * W[index] ) + l[index] )
#    return value[-1]

#LayersT = theano.shared(Layers)

#output, updates = theano.scan(
#            op,
#            sequences=x,
#            non_sequences=[LayersT, p],
#            truncate_gradient=4, )

#predict = theano.function([x], output)

#print predict(Inputs[0])

#def op(x):
#    value = []
#    for index in range(nLayers):
#        #value.append( T.dot( Functions[index][0]( x * W[index] ),p[index]) + l[index] )
#        value.append( Functions[index][0]( x * W[index] ) + LayersT[index] )
#    return value

#LayersT = theano.shared(Layers)

#output, updates = theano.scan(
#            op,
#            sequences=x,
#            truncate_gradient=4, )

#predict = theano.function([x], output)

#print predict(Inputs[0])

#print predict(np.array(np.linspace(0,1,num = 1), dtype = float))

# To sum up
# Theano function and scan are not very flexible. 
# Theano doesn't work very well with types different that Tensor Variables or TensorShared Variables.
# To work with the own variables in Theano, this must use the especific functions in order to perate with symbolic data like,
#  ifelse, where, ld, eq, et... since they does´t exist at first they cannnot resolve in compliation time.


##################
###  EXAMPLE 5
##################

#import theano
#import theano.tensor as T
#import numpy as np

#x = T.vector()

#nLayers = 3
##Create the input
#inputs = np.array(np.linspace(0,8,num = 9), dtype = float).reshape((3,3))  

#s =T.ones(3)

#def op(x,s):
#    cs = s + 4
#    o = T.sum(cs)

#    return [o , cs]

#output, updates = theano.scan(
#            op,
#            sequences=x,
#            truncate_gradient=4, 
#            outputs_info=[None, 
#                          dict(initial=s)])

#predict = theano.function([x], output)

#print predict(inputs[0])
       
##[array([ 15.,  27.,  39.]), array([[  5.,   5.,   5.],
##       [  9.,   9.,   9.],
##       [ 13.,  13.,  13.]])]

# #    o            cs
# #   15.,  [  5.,   5.,   5.]  x = 0
# #   27.,  [  9.,   9.,   9.]  x = 1
# #   39.,  [ 13.,  13.,  13.]  x = 2



##################
###  EXAMPLE 6
##################

#import theano
#import theano.tensor as T
#import numpy as np

#x = T.vector()

#nLayers = 3
##Create the input
#inputs = np.array(np.linspace(0,8,num = 9), dtype = float).reshape((3,3))  

#def op(x,s1, s2):

#    a = s1 + 1
#    b = s2 + 4
   
#    oa = T.sum(a)
#    ob = T.sum(b)
#    o = oa+ob

#    return [o , a, b]

#s = [T.ones(3),T.ones(4)]

#output, updates = theano.scan(
#            op,
#            sequences=x,
#            truncate_gradient=4, 
#            outputs_info=[None, 
#                          dict(initial=s[0]),
#                          dict(initial=s[1])])

#predict = theano.function([x], output)

#print predict(inputs[0])

##################
###  EXAMPLE 7
##################
#import theano
#import theano.tensor as T
#import numpy as np

#x = T.vector()

#nLayers = 3
##Create the input
#inputs = np.array(np.linspace(0,8,num = 9), dtype = float).reshape((3,3))  

#def op(x,s1, s2):

#    a = s1 + 1
#    b = s2 + 4
   
#    oa = T.sum(a)
#    ob = T.sum(b)
#    o = oa+ob

#    return [o , a, b]

#s = [T.ones(3),T.ones(4)]

#output, updates = theano.scan(
#            op,
#            sequences=x,
#            truncate_gradient=4, 
#            outputs_info=[None, 
#                          dict(initial=s[0]),
#                          dict(initial=s[1])])

#predict = theano.function([x], output)

#print predict(inputs[0])


#w = [] 
#w.append(T.ones(3))
#w.append(T.ones(4))

#def op2(x,w, w2):

#    w[0] = w[0] + 1
#    w[1] = w[1] + 4
   
#    oa = T.sum(w[0])
#    ob = T.sum(w[1])
#    o = oa+ob

#    return o

#output2, updates2 = theano.scan(
#            op2,
#            sequences=x,
#            non_sequences = w,
#            truncate_gradient=4)
          
#predict2 = theano.function([x], output2)

#print predict2(inputs[0])


##################
###  EXAMPLE 7
##################

#https://github.com/Theano/Theano/issues/3760
#The return value of the step function cannot contain nested 
#lists, which is what you have, just a flat list of variables.

import theano
import theano.tensor as T
import numpy as np

x = T.vector()

nLayers = 3
#Create the input
inputs = np.array(np.linspace(0,8,num = 9), dtype = float).reshape((3,3))  

def op(x,*ps):

    print ps[0][1]

    a = ps[0] + 1
    b = ps[1] + 4

    #ps = [state for state in ps] 
    #print ps[0]
    s = []
    oa = T.sum(a)
    ob = T.sum(b)
    o = oa+ob

    # sTore all the out puts in a list to return the values to the scan function
    s.append(o)
    s.append(a)
    s.append(b)
  
    return s

s = [] 
s.append(None)
s.append(dict(initial=T.ones(3)))
s.append(dict(initial=T.ones(4)))

#output, updates = theano.scan(
#            op,
#            sequences=x,
#            truncate_gradient=4, 
#            outputs_info=[None, 
#                          dict(initial=T.ones(3)),
#                          dict(initial=T.ones(4))])

output, updates = theano.scan(
            op,
            sequences=x,
            truncate_gradient=4, 
            outputs_info=s)

predict = theano.function([x], output)

print predict(inputs[0])


#import theano
#import theano.tensor as T
#import numpy as np

#x = T.vector()

#nLayers = 3
##Create the input
#inputs = np.array(np.linspace(0,8,num = 9), dtype = float).reshape((3,3))  

#def op(x,s1, s2):

#    a = s1 + 1
#    b = s2 + 4
#    s = []
#    s.append(a)
#    s.append(b)
   
#    oa = T.sum(a)
#    ob = T.sum(b)
#    o = oa+ob

#    return [o , np.asarray(s)]

#s = [] 
#s.append(T.ones(3))
#s.append(T.ones(4))

#e = np.asarray(s)

#output, updates = theano.scan(
#            op,
#            sequences=x,
#            truncate_gradient=4, 
#            outputs_info=[None, 
#                          dict(initial=e , taps=[-2,-1])])

#predict = theano.function([x], output)

#print predict(inputs[0])

# With shared parámeters doesn`t work because the weights are not updating 
#w = [theano.shared(np.ones((3,))),theano.shared(np.ones((4,)))]

#def op2(x):

#    w[0].set_value(w[0].get_value() + 1)
#    w[1].set_value(w[1].get_value() + 4)
   
#    oa = T.sum(w[0])
#    ob = T.sum(w[1])
#    o = oa+ob

#    return o

#output2, updates2 = theano.scan(
#            op2,
#            sequences=x,
#            outputs_info=None,
#            truncate_gradient=4)
          
#predict2 = theano.function([x], output2)

#print predict2(inputs[0])
