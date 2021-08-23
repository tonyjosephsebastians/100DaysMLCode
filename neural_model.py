
import numpy as np

input = np.array([[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]])
label = np.array([0,0,1,1,0,1,0,1])
label = label.reshape(8,1)
print(f"{input} {label}")

#Define hyperparameters
np.random.seed(42)
weights = np.random.rand(3,1)
bias = np.random.rand(1)
lr = 0.001

print(weights)
#Defie activation function

def sigmoid(x):
    return 1/(1+np.exp(-x))


#define partial derivative of sigmoid function

def sigmoid_partial(x):
    return sigmoid(x) * (1- sigmoid(x))

#Trainthe model
for epocs in range(25000):

    inp = input
    xw = np.dot(inp,weights) + bias
    z = sigmoid(xw)
    error = z - label
    print(error.sum())
    d_cost_pred = error
    d_pred_d_z = sigmoid_partial(z)
    z_del  = d_cost_pred * d_pred_d_z
    inputs_trans = input.T
    weights = weights - lr*np.dot(inputs_trans,z_del)

    for num in z_del:
        bias = bias - lr*num


#prediction

x = np.array([1,1,1])
pred = sigmoid(np.dot(x,weights) + bias)
print(pred)


