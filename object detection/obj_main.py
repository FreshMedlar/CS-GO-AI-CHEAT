# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 15:15:49 2023

@author: stefa
"""
from objDetectionDataset import ObjDetectionDataset
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm
from torch import nn
import torch.optim
import cv2
import numpy as np
import os

# Script for training the object detection model
def train(train_data_loader, model, optimizer, train_loss_list, train_itr):
    
     # initialize tqdm progress bar
    prog_bar = tqdm(train_data_loader, total=len(train_data_loader))
    
    for i, data in enumerate(prog_bar):
        optimizer.zero_grad()
        images, targets = data
        images = list(image.to("cuda") for image in images)
        targets = [{k: v.to("cuda") for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        train_loss_list.append(loss_value)
        losses.backward()
        optimizer.step()
        train_itr += 1
    
        # update the loss value beside the progress bar for each iteration
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
    return train_loss_list, train_itr

def collate_fn(batch):
    return tuple(zip(*batch))

train_dataset = ObjDetectionDataset("./data/images/", "./data/labels/")
# The collate_fn function is needed in order to normalize the input size of every item. 
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn = collate_fn)

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features
# replace the pre-trained head with a new one
num_classes = 3
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model = model.to("cuda")

params = [p for p in model.parameters() if p.requires_grad]
# Define criterion, optimizer, and scheduler
criterion = nn.CrossEntropyLoss()  # standard crossentropy loss for classification
# Optimizer initialization. We choose SGD and assing a momemntum of 0.9. The momentum help
# the SGD to correct the weights in the right direction faster. 
# Weight decay helps the model to keep weight small and avoid overfitting.
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
# decay the learning rate over time  helps the network converge to a local minimum and avoid oscillation
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

train_itr = 1
train_loss_list = []

for epoch in range(25):
    print(f"\nEPOCH {epoch+1} of 25")

    train_loss, train_itr = train(train_dataloader, model, optimizer, train_loss_list, train_itr)
    
torch.save(model.state_dict(), "./model.pt")