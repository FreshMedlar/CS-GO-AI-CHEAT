# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 09:29:55 2023

@author: stefa
"""

import os

# Script for eliminating taken images that have no bodies or heads in them
labels_path = "./data/labels/"
for img_name in os.listdir("./data/images"):
    if (f"{img_name.split('.')[0]}.txt" not in os.listdir(labels_path+"body")) and (f"{img_name.split('.')[0]}.txt" not in os.listdir(labels_path+"head")):
        os.remove(f"./data/images/{img_name}")