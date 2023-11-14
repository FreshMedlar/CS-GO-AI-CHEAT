# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 15:32:33 2023

@author: stefa
"""

import os
import random

def split_images(dataset_name, images_labels, dir_len, dir_name):
    
    # Pick a random image and delete it from the dictionary.
    # Then, move the images in the correct directory.
    d = dict()
    k = len(images_labels)-1
    for i in range(dir_len):
        index = random.randint(0,k)
        
        label = list(images_labels)[index]
        
        d[label] = images_labels[label]
        
        k = k - 1
        del images_labels[label]
        
    print(f"\n {dir_name.upper()}")
    print(f"Lunghezza richiesta: {dir_len}")
    print(f"Lunghezza attuale: {len(d)}")
    
    for k,v in d.items():
        os.rename(f"./{dataset_name}/{k}", f"./{dataset_name}/{dir_name}/labels/{k}")
        os.rename(f"./{dataset_name}/{v}", f"./{dataset_name}/{dir_name}/images/{v}")
        
    return images_labels

# The script take as input the directory name and split eveything in train, test and validation
# dataset. The directory should contain every image and the corresponding labels in a txt file
def main(dataset_name):
    
    # Create the directory structure needed for this program to work
    os.mkdir(f"./{dataset_name}/test/images/")
    os.mkdir(f"./{dataset_name}/test/labels/")
    
    os.mkdir(f"./{dataset_name}/train/images/")
    os.mkdir(f"./{dataset_name}/train/labels/")
    
    os.mkdir(f"./{dataset_name}/valid/images/")
    os.mkdir(f"./{dataset_name}/valid/labels/")
    
    l = os.listdir(os.path.join(".",dataset_name,"images"))
    
    # Split the images between text files and png files
    images = [i for i in l if i.split(".")[-1] == "png"]
    labels = [i for i in l if i.split(".")[-1] == "txt"]
    
    # Create a dictionary. For each label, it contains the corresponding image
    images_labels = dict()
    for i in images:
        for l in labels:
            name = i.split(".")[0]
            
            if name == l.split(".")[0]:
                images_labels[l] = i
           
    # Split the images using those percentage
    train = int(0.7 * len(images_labels))
    test = int(0.15 * len(images_labels))
    valid = int(0.15 * len(images_labels))
        
    # Split the directory and move the images and labels
    image_labels = split_images(dataset_name, images_labels, train, "train")
    image_labels = split_images(dataset_name, images_labels, test, "test")
    image_labels = split_images(dataset_name, images_labels, valid, "valid")
    
    return images_labels

if __name__ == "__main__":
    main()