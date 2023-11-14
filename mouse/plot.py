# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 09:40:33 2023

@author: stefa
"""

import matplotlib.pyplot as plt
from data_loader import load_data

# For each path, we take the x and y values and return them as lists.
def get_points(path):
    x = []
    y = []

    for point in path:
        x.append(point[0])
        y.append(point[1])
        
    return x,y

# Plot the given paths
def plot_paths(paths):
    
    for path in paths:
        x,y = get_points(path)
        plt.plot(x,y)
        
    plt.show()
