""" Given a flow output to the images. it saves all rgb.npy and flow.npy files to input to evalutate_sample.py
"""

import numpy as np
import cv2
import glob, os
from utils import *


images = glob.glob("vid/Chicago_Amanda_S_LivingRoom_20150110100356/img_*.jpg")
images.sort()
flow_x = glob.glob("vid/Chicago_Amanda_S_LivingRoom_20150110100356/flow_x_*.jpg")
flow_x.sort()
flow_y = glob.glob("vid/Chicago_Amanda_S_LivingRoom_20150110100356/flow_y_*.jpg")
flow_y.sort()
print(len(images), len(flow_x), len(flow_y))


resize_len=256
crop_w = 224
crop_h = 224

rgb = read_vid_rgb(images, resize_len, crop_h, crop_w)
rgb = np.expand_dims(rgb, axis=0)
rgb = np.float32(rgb)

for c in range(rgb.shape[4]):
    tmp = np.float32(rgb[:,:,:,:,c])
    rgb[:,:,:,:,c] = ((tmp - np.min(tmp)) / (np.max(tmp) - np.min(tmp)) - 0.5) * 2

#rgb=rgb[:, :79]
print(rgb.shape)
#print(rgb)
np.save("vid/rgb",rgb)

flow = read_vid_flow(flow_x, flow_y, resize_len, crop_h, crop_w)
flow = np.expand_dims(flow, axis=0)
flow = np.float32(flow)

for c in range(flow.shape[4]):
    tmp = np.float32(flow[:,:,:,:,c])
    flow[:,:,:,:,c] = ((tmp - np.min(tmp)) / (np.max(tmp) - np.min(tmp)) - 0.5) * 2

#flow = flow[:, :79]
print(flow.shape)
#print(flow)
np.save("vid/flow",flow)
