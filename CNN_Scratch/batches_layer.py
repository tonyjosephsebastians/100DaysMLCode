import numpy as np

x = [[1,2,3,4],
    [2,3,4,5],
    [1.5,2,3.6,4]]

class layer_Dense:

    def __init__(self,n_inputes,n_neurones):
        self.weights = np.random.randn(n_inputes,n_neurones)
        self.bias = np.random.randn(1,n_neurones)

    def forward(self,n_inputs):
        self.output = np.dot(n_inputs,self.weights) + self.bias



class activation_RELU:

    def forward(self,n_inputs):
        self.output = np.maximum(0,n_inputs)


    
layer1 = layer_Dense(4,5)
layer2 = layer_Dense(5,2)

layer1.forward(x)
print(layer1.output)

layer2.forward(layer1.output)

print(layer2.output)

activation = activation_RELU()
activation.forward(layer2.output)
print(activation.output)

