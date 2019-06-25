# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 18:07:06 2019

@author: Dell
"""
import numpy as np
def computecost(X, y, theta):
    J = 0
    m = np.size(y)
    h = np.dot(X, theta)
    diff = h-y
    J = (1/(2*m)) * np.sum(np.power((diff),2))
    return J