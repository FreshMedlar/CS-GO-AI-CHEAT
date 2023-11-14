# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 18:04:08 2023

@author: stefa
"""

import json
import numpy
import pandas
import matplotlib.pyplot as plt

# load the data
def load_data(file_path_paths):
    
    # Load JSON file
    data = load_data_from_json(file_path_paths)

    # List of dictionary. Each dictionary contains time, x and y value for a mouse trajectory
    paths = pandas.DataFrame(data).values

    # List for values
    l_path = []
    l_destination = []
    l_time = []

    # Since the data are in the format [(x_values, y_values, time_values)] for each path,
    # we take them, zip into a dictionary and append the last value (the destinatio point) to
    # the right list
    for path in paths:
        x = path[0]
        y = path[1]
        t = path[2]
        last_x = x[-1]
        last_y = y[-1]
        l_destination.append([last_x, last_y])

        a = list(zip(x, y))
        l_path.append(a)
        l_time.append(list(zip(t)))

    # Parsing the value to numpy arry
    l_path = numpy.array(l_path)
    l_destination = numpy.array(l_destination)
    l_time = numpy.array(l_time)
    return l_destination, l_path, l_time


# Open the json 
def load_data_from_json(file_path):
    with open(file_path) as json_file:
        data = json.load(json_file)

    return data
