# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 09:31:03 2023

@author: stefa
"""
import ctypes
import win32gui
import torch
import numpy as np
from PIL import ImageGrab
import cv2
import traceback

import sys
sys.path.insert(1, "./mouse/")
from mouse_mover import MouseMover

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

class VideoAnalyzer():
    
    # Class constructor
    def __init__(self, class_name_dict = {0:"head", 1:"body"}):
        ctypes.windll.shcore.SetProcessDpiAwareness(1) # Without it video acquisition won't work
        
        self.winlist = [] # Needed for window acquisition
        self.window = None # Window to be acquired
        self.model = None # Model to be used on video frames
        self.class_name_dict = class_name_dict # Image recognition classes
        self.mouse_mover = MouseMover() # Class for mouse movement (human-like aimbot)
        
    # This method analyze the video feed and open a window in which the frame with bounding boxes
    # for objects are visualized
    def analyze_video_feed(self, scale_percent=60, threshold=0.8):
        while True:
            try:
                # Grab the frame from the window
                img = np.array(ImageGrab.grab(bbox=win32gui.GetWindowRect(self.window)))
                
                # Analyze the frame getting as result the analyzed frame with bounding boxes
                img = self.analyze_frame(img, threshold)
                
                # Dimension to scale the analyzed frame
                width = int(img.shape[1] * scale_percent / 100)
                height = int(img.shape[0] * scale_percent / 100)
                dim = (width, height)
                  
                # Resize image
                resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
                
                # Show the analyzed frames
                cv2.imshow('window',cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break
            except:
                print(traceback.format_exc())
          
    # Main point from which the frames are analyzed and eventual mouse movements
    # are performed
    def analyze_frame(self, img, threshold):
        
        # Copying the original image to draw on it later the bounding boxes
        orig_img = img.copy()
        
        # Perform some operation in order to be able to pass the frame to the model
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0 # The model works with values between 0 and 1, so we first normalize them
        img = np.transpose(img, (2, 0, 1)) # Transpose the image from [W,H,C] format to [C,W,H] as the model requires
        img = torch.tensor(img, dtype=torch.float)
        img = img.unsqueeze(0).to("cuda") # The model expect the image to have shape like [B,C,W,H] where B is the batch size.
                                          # Since it's a single image, B will be one by unsqueezing the first axis.
        
        results = self.model(img) # Get the predictions
        
        to_shoot = dict() # Dictionary in which we are gonna insert points in where to shoot with corresponding score
                          # given by the model and class id
        
        # Take the results back to the cpu because the operations are not done with cuda loaded tensors.
        results = [{k: v.to('cpu') for k, v in t.items()} for t in results] 

        # The result from the model is a list of dictionary, so we take the corresponding values
        boxes = results[0]['boxes'].data.numpy()
        scores = results[0]['scores'].data.numpy()
        labels = results[0]['labels'].data.numpy()

        # Loop through bounding boxes
        for j, box in enumerate(boxes):
            # For each bounding box we take the score and label
            score = scores[j]
            class_id = labels[j]
            
            # If the score is higher than a certain threshold we create the needed values and frame with bounding boxes
            if score > threshold:
                cv2.rectangle(orig_img,(int(box[0]), int(box[1])),(int(box[2]), int(box[3])),(0, 0, 255), 2)
                    
                # Compute the values of center points in bounding boxes. Those are the points in which 
                # we are gonna aim. We finally draw and save them in the previously created dictionary 
                mid_x = int((box[0]+box[2])/2)
                mid_y = int((box[1]+box[3])/2)
                cv2.circle(orig_img, (mid_x, mid_y), radius=1, color=(255,0,0), thickness=10)
                
                to_shoot[mid_x, mid_y] = (score, class_id)
                    
        # After frame analyzation we set in the mouse mover the points in where to shoot
        self.mouse_mover.set_targets(to_shoot)
        
        # After setting the points in where to shoot, we perform the shooting action
        self.mouse_mover.shoot()
        
        return orig_img
    
    # Initialize the model for object detection
    def init_model(self, path_to_weights):
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        num_classes = 3
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        self.model.load_state_dict(torch.load(path_to_weights))
        self.model = self.model.to("cuda")

        self.model.eval()
        
    # Get the window in which the game is visualized
    def get_window(self, window_name):
        self.window = self.get_screens(window_name)[0][0]
        print("[+] WINDOW ACQUIRED")
        
    # ====== (DON'T TOUCH) METHODS FOR WINDOW ACQUISITION (DON'T TOUCH) ======
    def enum_cb(self, hwnd, results):
        self.winlist.append((hwnd, win32gui.GetWindowText(hwnd)))
        
    def get_screens(self, screen_name):
        win32gui.EnumWindows(self.enum_cb, self.winlist)
        screens = [(hwnd, title) for hwnd, title in self.winlist if screen_name.lower() in title.lower()]
        while len(screens) == 0:
            screens = [(hwnd, title) for hwnd, title in self.winlist if screen_name.lower() in title.lower()]
            win32gui.EnumWindows(self.enum_cb, self.winlist)

        return screens
        