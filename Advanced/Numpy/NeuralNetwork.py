import Statistics
import numpy as np

# Class Nural Network
class NeuralNetwork (object):
      
    # Shape and functions used for the neural network.
    Shape = [] # [3,5,1]  
    Functions = [] # [None, Functions.sigmoid, Functions.linear]
   
    # Constructor for the class
    def __init__(self,shape,functions, initialize = False):
        if (len(shape) != len(functions)):
            raise Exception("NeuralNetwork", "Layers and functions length doesn't match." )
        self.Shape = shape
        self.Functions = functions
         # Check if it must be initializedin the creation
        if (initialize):
            self.Init() # Initialize with the default settings (bias = 0, Random)

    # Initialize function
    def init(self, use_bias = False, mode = "R"):
        """Initialize the Neural Network with the shape configured in the constructor and using the mode.

        Parameters
        ----------
        mode : {'R', 'Z', 'O','G'}, optional
           The way to initalize the weigths for the units:
                'R': Random numbers from (0,1)
                'G': Gaussian Random generation
                'Z': Set ceros numbers
                'O': Set ones for the initialization numbers
     
        """
        # Set if a Bias must be used for computing 
        self.use_bias = use_bias
        bias = 1 if use_bias else 0   # 1 si use bias -> else -> 0
        # NOTE: While the training the BIAS used in the inputs will be 1, However the weights will be updated by the train method used.
        
        # Crate all the layers (inputs, outputs and weigths)
        self.inputs = []
        self.outputs = []
        self.weigths = []

         # Crate all the weigths (layer k - 1 ->  layer K, using the shape of the network)
         # For a Neutal Network with Share (3,5,1) with N inputs for tranining and testing 
         #          INPUT LAYER             HIDDEN LAYER             OUTPUT LAYER
         #             N X 3                    N x 5                    1 x N
         #                       3 x 5                      5 x 1 
         #                        W(1)                      (W2)
         #  * If BIAS additional unit will be added to each layer and weight  
        for wsizes in zip(self.Shape, self.Shape[1:]):  #Take the next element of the array and use the zip to join both together
            #Check if it's the last weight so not include the bias term
            if (mode == 'G'):
                self.weigths.append(np.random.normal(scale=0.01, size = (wsizes[1] + bias,wsizes[0] + bias)).T) #Create as transposes to do the dot function
            elif (mode == 'Z'):
                self.weigths.append(np.zeros((wsizes[1] + bias,wsizes[0] + bias)).T) 
            elif (mode == 'O'):
                self.weigths.append(np.ones((wsizes[1] + bias,wsizes[0] + bias)).T)
            else:
                self.weigths.append(np.random.rand(wsizes[1] + bias,wsizes[0] + bias).T)

        #Remove the last columns with the bias term from the weights of the last layer
        if (self.use_bias):
            self.weigths[-1] = np.asarray(self.weigths[-1])[:,:-1]
    
    def forward(self, inputs):
        """Forward propagation fromt inputs to inputs using current weights and bias
        Parameters
        ----------
        inputs : inputs to be run into the Neural Network
        
        Returns
        -------
        out : ndarray
            Array with the predicted outputs using the inputs given in the call function.
        """
        #Convert the inputs and outputs to numpy arrays
        inputs = np.asarray(inputs, dtype = np.float)
         
        # Check if inputs and outputs have the same size as the originally configured (without bias units)
        if (inputs.shape[1] != self.Shape[0]):
            return -1
        
        self.inputs = [] 
        # Set the new inputs into the network (units x observations) -> If bias additional unit will be added set to 1
        if (self.use_bias):
            self.inputs.append(np.append(inputs, np.ones((inputs.shape[0],1)),axis = 1)) # Add a new column with the bias as additional unit
        else:
            self.inputs.append(inputs)
       
        self.outputs = [] 
        # Set the outputs in the first layer the same as the inputs since there are no functions nor activation
        self.outputs.append(self.inputs[0])
                
        # Propagate the inputs over the current neural network
        for index in range(1,len(self.Shape)): 
            #Compute the dot product and set into the current hidden layer
            self.inputs.append ( np.dot(self.outputs[index-1],self.weigths[index-1]))
            #Compute the activation Function and set to te next output
            self.outputs.append ( self.Functions[index](self.inputs[index]))

        # Return the last layer activation with the predected outputs
        return self.outputs[-1]
     
    def printLayers(self):
        #Print the entire neural Network
        for index in range(0,len(self.Shape)): 
            print ("Layer " , index)
            print ("  Inputs")
            print(self.inputs[index])
            print ("  Outputs")
            print(self.outputs[index])
            if (index < len(self.Shape) - 1):
                print ("  Weigths")
                print(self.weigths[index])