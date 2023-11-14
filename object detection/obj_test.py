# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 13:07:38 2023

@author: stefa
"""

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch
import cv2
import numpy as np

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features
# replace the pre-trained head with a new one
num_classes = 2
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model.load_state_dict(torch.load("./model.pt"))
model = model.to("cuda")

model.eval()

img_name = "image.png"
orig_img = cv2.imread(img_name)
img = np.array(orig_img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img / 255.0
img = np.transpose(img, (2, 0, 1))
img = torch.tensor(img, dtype=torch.float)
img = img.unsqueeze(0).to("cuda")

y = model(img)
print(y)
y = [{k: v.to('cpu') for k, v in t.items()} for t in y]

boxes = y[0]['boxes'].data.numpy()

for j, box in enumerate(boxes):
    cv2.rectangle(orig_img,(int(box[0]), int(box[1])),(int(box[2]), int(box[3])),(0, 0, 255), 2)

cv2.imshow("window", orig_img)
cv2.waitKey(0)
