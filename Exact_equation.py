# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 18:20:20 2019

@author: Dell
"""
import numpy as np
def normal_solution(X, y):
    theta_ex = np.dot(np.dot(np.linalg.pinv(np.dot(X.T, X)), X.T), y)
    return theta_ex
