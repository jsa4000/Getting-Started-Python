import Statistics
from NeuralNetwork import NeuralNetwork
import numpy as np

# Class Back Propagation
class BackPropagation (object):
          
    # Constructor for the class
    def __init__(self,neural_network, trainingRate = 0.25, momentum = 0.15 ):
       # Set the current Network to Train
       self.NN = neural_network
       self.trainingRate =trainingRate
       self.momentum = momentum
    
    # Initialize function
    def train(self, inputs, outputs, maxError = 0.0001, maxIterations = 100000):
        #Initialize previous delta
        self.previousWeightDelta = []
        
        NN = self.NN
        bias = 1 if NN.use_bias else 0   # 1 si use bias -> else -> 0
        for wsizes in zip(NN.Shape, NN.Shape[1:]):  #Take the next element of the array and use the zip to join both together
                self.previousWeightDelta.append(np.zeros((wsizes[1] + bias,wsizes[0] + bias)).T)
           
        #Remove the last columns with the bias term from the weights of the last layer
        if (NN.use_bias):
            self.previousWeightDelta[-1] = np.asarray(self.previousWeightDelta[-1])[:,:-1]      
          
        # Train the Network with given the inputs and outputs
        for iter in range(0, maxIterations):
            # Call update() -> Forward propagation + update weights and Bias
            error = self.update(inputs,outputs)
            # Get the current Error (Const function) implemented by the trainer
            if iter % 5000 == 0 and iter > 0:
                 print("Iteration {0:6d}K - Error: {1:0.6f}".format(int(iter/1000), error))
            if error <= maxError:
                print("Desired error reached. Iter: {0}".format(iter))
                break

        return iter

    def getError (self, toutputs, outputs):
         # Case of being back-propagating from the last layer
        output_delta = (toutputs - outputs)
        return np.sum(output_delta**2) # Get the error using the least squares function

    def update(self, inputs, outputs):
        """Backprpagation algorithm inputs to inputs using current weights and bias
        Parameters
        ----------
        inputs : inputs to be run into the Neural Network
        outputs : expected output to compute the error and perform the backpropagation in order to update the weights
        
        Returns
        -------
        out : float
            The error (cost functions) for the current configuration using least squares
        """

         #Get the current neural network
        NN = self.NN
        layerCount = len(NN.Shape) - 1

         #Convert the inputs and outputs to numpy arrays
        inputs = np.asarray(inputs, dtype = np.float)
        outputs = np.asarray(outputs, dtype = np.float)
      
        # Check if inputs and outputs have the same size as the originally configured (without bias units)
        if (inputs.shape[1] != NN.Shape[0]):
            return -1
        if (outputs.shape[1] != NN.Shape[-1]):
            return -1
       
        # Call Forward propagation with the following inputs
        NN.forward(inputs)
        # Store the delta for the derivatives 
        delta = []
        error = 0

        # Back propagate the network from the outputs to the inputs
        #for index in range(len(NN.Shape)- 1, 0, -1):  
        for index in reversed(range(layerCount)):
            # Check the current layer
            if (index == layerCount - 1):
                # Case of being back-propagating from the last layer
                output_delta = (NN.outputs[-1] - outputs)
                error = np.sum(output_delta**2) # Get the error using the least squares function
                delta.append(output_delta * NN.Functions[index+1](NN.inputs[index + 1], True))
            else:
                # Case of being back-propagating from hidden layers
                #delta_pullback = NN.weigths[index+1].dot(delta[-1].T) # Because vectorization
                delta_pullback = delta[-1].dot(NN.weigths[index+1].T) # Because vectorization
                delta.append(delta_pullback * NN.Functions[index+1](NN.inputs[index+1], True))
      
        #Reverse the list 
        delta.reverse()

        #Update the Weights
        for index in range(layerCount ):
            # Compute the current weight delta derivative 
            curWeightDelta = np.dot(NN.outputs[index].T, delta[index])
            # Compute the weight delta using the previous one obtained to apply momentum to increase the speed of backpropagation-
            # Momentum speed up the process when the steps and weights are almost identical. An example is like a ball falling down a hill.
            # On the other hand if momentum is very high the acceleration is too much and the graident can distant so much from the local minima. 
            weightDelta = self.trainingRate * curWeightDelta + self.momentum * self.previousWeightDelta[index]
            # Substract the previous weigh with this one.
            NN.weigths[index] -= weightDelta
            self.previousWeightDelta[index] = weightDelta

        return error
   
        