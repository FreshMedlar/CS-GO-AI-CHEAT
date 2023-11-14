# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 11:19:31 2023

@author: stefa
"""

from torch.utils.data import Dataset
import os
import numpy as np
import cv2
import torch

# Custom dataset for object detection
class ObjDetectionDataset(Dataset):

    # Initialize the images path and labels path
    def __init__(self, images_path, labels_path, transforms=None):
        self.images_name = os.listdir(images_path)
        self.images_path = images_path
        self.labels_path = labels_path


    def __len__(self):
        return len(self.images_name)

    # Take the image from the path provided
    def __getitem__(self, index):
        
        # Perform some transformation in order to be able to 
        # pass the image to the model
        img_name = self.images_name[index]
        img = cv2.imread(self.images_path + img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.array(img)
        img = img / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = torch.tensor(img, dtype=torch.float)

        # Create the dictionary that sould be passed during the 
        # training phase to the model
        target = {}
        target["image_id"] = int(img_name.split('.')[0])
        target["boxes"] = []
        target["area"] = []
        target["labels"] = []
        target["iscrowd"] = []


        target = self.create_target(img_name, "body", target)
        target = self.create_target(img_name, "head", target)

        target["image_id"] = torch.tensor(target["image_id"])
        target["boxes"] = torch.tensor(target["boxes"])
        target["area"] = torch.tensor(target["area"])
        target["labels"] = torch.tensor(target["labels"])
        target["iscrowd"] = torch.tensor(target["iscrowd"])

        return img, target

    def create_target(self, img_name, label_name, target = dict()):
        if f"{img_name.split('.')[0]}.txt" in os.listdir(self.labels_path+label_name):
            f = open(self.labels_path + label_name + "/" + f"{img_name.split('.')[0]}.txt")
            lines = f.readlines()
            for line in lines:
                x1, y1, x2, y2 = line.split()
                x1 = float(x1)
                y1 = float(y1)
                x2 = float(x2)
                y2 = float(y2)
                target["boxes"].append([x1,y1,x2,y2])

                area = (x2-x1) * (y2-y1)
                target["area"].append(area)

                c = -1
                if label_name == "body":
                    c = 1
                else:
                    c = 2
                target["labels"].append(c)

                target["iscrowd"].append(False)
        return target
     
    