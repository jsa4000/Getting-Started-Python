import numpy as np
import Statistics
import DataSet
import Plotter
from NeuralNetwork import NeuralNetwork
from BackPropagation import BackPropagation

def main():
    # Start the code
    
    # Get the inputs and expected outputs to train the Neural Network
    inputs, outputs = DataSet.getDataSet( ".\Data\data_sets.xml", 0)
    tinputs, toutputs = DataSet.getDataSet( ".\Data\data_sets.xml", 0)

    tinputs =  np.array([[0, 0], [1, 1], [0, 1], [1, 0]])
    toutputs = np.array([[0.00], [0.00], [1.00], [1.00]])
    
    # Set the Number of layers and units 
    layers = [2,5,1] # This will create a neural network with 3 layers with, 2 inputs, one hidden layer with 5 units and an output layer with 2 units. (dataset 2)
    #layers = [25,5,4] #dataset 1

    # Set the activation functions per layer    
    activations = [None, Statistics.tanh, Statistics.linear]

    #Create and train the Neural Network
    try:
        
        # Create the Neural network with the parameters previoulsy configured
        nn = NeuralNetwork(layers, activations, initialize = False)
        
        #Initialize Neurak Network with the weights
        nn.init(True, "R")
      
        #Create the trainer that will train the network using the train method and the parameters
        trainer = BackPropagation(nn,trainingRate = 0.25, momentum = 0.001)

        # Train the Network
        iterations = trainer.train(tinputs,toutputs,maxError = 1e-6, maxIterations = 50000)
        print ("Iterations: ",iterations)

        # Print Current Trained Network
        poutputs = nn.forward(tinputs)
        print(trainer.getError(poutputs,toutputs))
        print (poutputs)
        #nn.printLayers()
         
    except Exception as ex:
        # Error while creating the Neural Network
        print ex.args     # the exception instance


if __name__ == "__main__":
     # Main function
    main()
    