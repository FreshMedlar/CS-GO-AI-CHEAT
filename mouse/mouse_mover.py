# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 10:27:36 2023

@author: stefa
"""

import pyautogui
import mouse
import torch
from custom_mouse_model import MouseModel
import numpy
import cv2

# Class for mouse movement
class MouseMover():
    
    # Class constructor
    def __init__(self):
        
        # Set all the needed variables for the library in order to work in the proper way
        pyautogui.FAILSAFE = False
        pyautogui.MINIMUM_DURATION = 0
        pyautogui.MINIMUM_SLEEP = 0
        pyautogui.PAUSE = 1e-10 # Minimum duration between each movement. The actual value is higher, but we set it to the lowest possible
        
        self.button_pressed = False # Check if the right button is pressed in order to perform the shooting action
        self.to_be_shot = None # List of points in which we want to aim
        self.model = MouseModel().to("cuda") # Create the model for the mouse movement
        self.model.load_state_dict(torch.load("./mouse/mouse_model_weights.pt")) # Load the weights for the mouse movement model
        
    # Set the targets
    def set_targets(self, to_be_shot):
        self.to_be_shot = to_be_shot
        
    # We select the best point based on object detection model confidence.
    # We take the actual mouse position and pass to the method that will perform the shooting
    def shoot(self):
        if mouse.is_pressed("right"):
            best_point = max(self.to_be_shot, key = lambda k: self.to_be_shot[k][0])
            current_point = pyautogui.position()
            self.parse_coordinates(best_point, current_point)
            
    # Method for mouse movement
    def parse_coordinates(self, to_reach, current_point):
        
        # Since the model assume our initial position as (0,0), we simply normalize
        # the coordinates. So, now the movement is performed from (0,0) to (distance_x, distance_y)
        print(f"To reach: {to_reach}")
        distance_x = to_reach[0] - current_point[0]
        distance_y = to_reach[1] - current_point[1]
        
        # Take the size of the screen
        size = pyautogui.size()
        
        # Since the model works with uniform values in [0,1], we need to normalize them using the screen size.
        # Also, the model expect the input shape to be like [B, D] where B is the batch size and D the destination point.
        # So, we unsqueeze it.
        x = torch.tensor([distance_x/size[0], distance_y/size[1]]).double().unsqueeze(0)
        
        # Got the result, we take the data and reshape the array to better perform the followinr computation
        y = self.model(x)
        y = y.cpu().detach().numpy()
        y = numpy.array(y).squeeze(0)

        # Since a 100 points path take forever to be traversed, we took just 20 points, one in every five.
        # After that, we take back the result to screen coordinates
        y = y[::5]
        y = [(i[0]*size[0], i[1]*size[1]) for i in y]
        
        # Those variables are needed for the last movement
        tot_x = 0
        tot_y = 0
        
        # After adding the points value to the total movements performed along the axes, 
        # we move relative to the previous mouse position. 
        # The relative movement is important because the 3D projection on screen change every time
        # we move the player perspective. Also, the 5.5 is an hardcoded value for the default sensibility of CS:GO (2.50),
        # if we don't divide for such value, the movement is gonna be completly offset.
        for i in y:
            tot_x = tot_x + i[0]/5.5
            tot_y = tot_y + i[1]/5.5
            pyautogui.moveRel(i[0]/5.5, i[1]/5.5, 0) 
            print(f"X: {i[0] + current_point[0]}, Y: {i[1] + current_point[1]}")
            
        # Since the final destination of the path is a predicted value, it will be slightly 
        # off from the actual destination point. In order to improve accuracy, we add a last
        # movement to center the aim to the actual destination point
        pos_x = current_point[0] + tot_x
        pos_y = current_point[1] + tot_y
        dx = to_reach[0] - pos_x
        dy = to_reach[1] - pos_y
        print(f"{dx}, {dy}")
        pyautogui.moveRel(dx, dy, pyautogui.easeInBounce)
        
        # While the button for mouse auto movement is pressed, we wait. This is done
        # to avoid strange flicks from one enemy to another
        while mouse.is_pressed("right"):
            pass
        
    