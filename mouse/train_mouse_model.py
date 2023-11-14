# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 18:25:45 2023

@author: stefa
"""

import torch
from torch import nn
from data_loader import load_data
from dataset import PathDataset
from torch.utils.data import DataLoader
from plot import plot_paths
from custom_mouse_model import MouseModel
    
# Script for training the mouse model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = MouseModel().to(device)

epochs = 1000
batch_size = 64
learning_rate = 1e-3

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

destinations, paths, times = load_data("./data.json")

train_dataset = PathDataset(destinations, paths)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=True)

def train_loop(dataloader, model, loss_fn, optimizer):
    
    for batch, (X,y) in enumerate(dataloader):
        pred = model(X)
        pred = pred.cuda()
        y = y.cuda()
        loss = loss_fn(pred, y)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if batch % 100 == 0:
            print(f"loss: {loss.item()}")
    
            
for t in range(epochs):
    print(f"Epoch: {t}")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    
print("Done")

torch.save(model.state_dict(), "./best.pt")

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        