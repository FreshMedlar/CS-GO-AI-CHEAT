# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 15:21:48 2023

@author: stefa
"""
from torch import nn

# Model for mouse movements predictions
class MouseModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, 1000), # Take as inputs the destination
            nn.ReLU(),
            nn.Linear(1000, 200), # Output 200 values
            nn.Unflatten(1, (100, 2)) # Create the 100 points by unflattening the output
        )
        self.double() # Parse the weights to be double instead of float. Done in order to avoid
                      # coordinates parsing to float every time
        
    def forward(self, x):
        out = self.mlp(x.cuda())
        return out