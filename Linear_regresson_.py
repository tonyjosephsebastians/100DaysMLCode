import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class LinearRegression:

    def __init__(self,iter =1000,learning_rate =0.001):

        self.iter =iter
        self.learning_rate = learning_rate
    
    def fit(self,x_train,y_train):
        self.b = 0
        self.x = x_train
        self.y = y_train
        self.theta = np.zeros(x_train.shape[1])

        for i in range(0,self.iter):
            self.update_weights()
        
            
    def predict(self,x):
        val = np.dot(x,self.theta) + self.b

        return val

    def update_weights(self):
            cost_list = []
            self.cost = cost_list
            y_pred = self.predict(self.x)
            cost = (1/(2*x.shape[0])) * np.sum(np.square(y_pred - y_train))
            self.cost.append(cost)
            d_theta = (1/(2 *x.shape[0])) * np.dot(self.x.T,y_pred - self.y)
            self.theta = self.theta - self.learning_rate * d_theta
            dbias = - 2 * np.sum( self.y - y_pred ) / self.x.shape[0]
            self.b = self.b - self.learning_rate * dbias


data = pd.read_csv('iris.csv')
x= data.iloc[:,:-1]
y =data.iloc[:,-1]
le = LabelEncoder()
y = le.fit_transform(y)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
print(x_train.shape)
model = LinearRegression()
model.fit(x_train,y_train)
x_pred = np.array([7,3,4,2])

pred = model.predict(x_pred)
print(np.round(pred))

