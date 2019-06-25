# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 08:08:29 2019

@author: Dell
"""
import numpy as np
from computeCost import computecost
def gradientdescent(X, y, theta, alpha, iterations):
    m = np.size(y)
    J = []
    for i in range(0,iterations):
        h = np.dot(X, theta)
        diff = h-y
        cost = computecost(X, y, theta)
        J.append(cost)
        theta = (theta.T-(alpha/m)*np.sum(np.multiply(diff, X), axis = 0)).T
    return theta,J
        