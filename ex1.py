# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 16:36:17 2019

@author: Dell
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from warmUp import warmUp
from computeCost import computecost
from gradientDescent import gradientdescent
from Exact_equation import normal_solution

identity_matrix = []
identity_matrix = warmUp()

data = 'ex1data1.txt'
dataSet = pd.read_csv(data, sep = ',', names = ["Area", "Price"])

X = dataSet.iloc[:,:-1].values
y = dataSet.iloc[:,-1].values
X = np.reshape(X,(np.size(X),1))
y = np.reshape(y,(np.size(y),1))

m = np.size(X)
X0 = np.ones((m,1))
X = np.concatenate((X0, X), axis = 1)

#Plotting the training set
plt.figure(1)
plt.scatter(X[:,-1],y, color = 'red', s = 7, alpha = 0.8)
plt.title('Training set')
plt.xlabel('Area')
plt.ylabel('Price')
#Gradient descent
theta = np.zeros((2,1))
iterations = 1500
alpha = 0.01
initial_cost = computecost(X, y, theta)
(theta, J) = gradientdescent(X, y, theta, alpha, iterations)

#Regression plot
plt.figure(2)
plt.scatter(X[:,-1],y, color = 'red', s = 7, alpha = 0.8)
plt.xlabel('Area')
plt.ylabel('Price')

h = np.dot(X, theta)
plt.plot(X[:,-1],h, color = 'blue')
plt.title('Regression fitting')
plt.xlabel('Area')
plt.ylabel('Price')

#Prediction
Prediction1 = np.dot([1, 3.5], theta)
Prediction2 = np.dot([1, 7], theta)

#3D Visualization
theta0_vals = np.linspace(-10, 10, 100);
theta1_vals = np.linspace(-1, 4, 100);
theta0_vals = np.reshape(theta0_vals,(np.size(theta0_vals),1))
theta1_vals = np.reshape(theta1_vals,(np.size(theta1_vals),1))


J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))
for i in range(0, len(theta0_vals)):
    for j in range (0, len(theta1_vals)):
        t = [theta0_vals[i], theta1_vals[j]]
        t = np.reshape(t,(np.size(t),1))
        J_vals[i][j] = computecost(X, y, t)

fig = plt.figure(3)
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(theta0_vals, theta1_vals, J_vals)
ax.set_title('Hypothesis function')
ax.set_xlabel('θ0')
ax.set_ylabel('θ1')
ax.set_zlabel('J(θ0,θ1)')
plt.show


Exact_theta = normal_solution(X, y)
