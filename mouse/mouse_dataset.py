# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 09:13:27 2023

@author: stefa
"""

from torch.utils.data import Dataset

# Dataset for mouse paths
class PathDataset(Dataset):
    def __init__(self, destinations, paths):
        self.destinations = destinations
        self.paths = paths
    
    def __len__(self):
        return len(self.destinations)
    
    def __getitem__(self, idx):
        return self.destinations[idx], self.paths[idx]
    
    