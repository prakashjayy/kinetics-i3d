""" Utils file
"""
import numpy as np
import cv2
import os
from tqdm import tqdm
import glob
import random


def one_hot(batch_LABELS):
    nb_classes = len(_LABELS)
    targets = np.array([_LABELS.index(i) for i in batch_LABELS])
    one_hot_targets = np.eye(nb_classes)[targets]
    return one_hot_targets

def random_crop(array):
    h = array.shape[2]
    w = array.shape[3]
    #print(h, w, array.shape)
    hw = int((h - 224)/2)
    wh = int((w - 224)/2)
    #print(hw, wh)
    num1 = np.random.randint(0,hw)
    num2 = np.random.randint(0,wh)
    #crop = [ik[ :, :,  num1:num1+224, num2:num2+224, :] for ik in array]
    return  array[:, :, num1:num1+224, num2:num2+224, :]

def center_crop(array):
    h = array.shape[2]
    w = array.shape[3]
    hw = int((h-224)/2)
    wh = int((w-224)/2)
    return array[:, :, hw:hw+224, wh:wh+224, :]

def read_vid_rgb(images, resize_len, crop_h, crop_w):
    """ read a list of images and concatenate them into numpy array
    """
    vid = []
    for i in images:
        img = cv2.imread(i)

        # convert color
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # resize
        h, w = img.shape[:2]
        resize_ratio = float(resize_len) / min(h, w)
        img = cv2.resize(img, None, fx=resize_ratio, fy=resize_ratio)

        # crop center
        h, w = img.shape[:2]
        center_y = int((h - crop_h)/2)
        center_x = int((w - crop_w)/2)
        #print(center_y, center_x, h, w, resize_ratio)
        img = img[center_y:center_y+crop_h, center_x:center_x+crop_w, :]
        vid.append(img[np.newaxis, :, :, :])
    vid = np.concatenate(vid)
    return vid


def generate_numpy_files(vid_img_loc, resize_len, crop_h, crop_w, batch_size=79):
    """Generates .npy files for each video at 79 frames per iteration iteratively.
    """
    images = glob.glob(vid_img_loc+"img_*.jpg")
    images.sort()
    flow_x = glob.glob(vid_img_loc+"flow_x_*.jpg")
    flow_x.sort()
    flow_y = glob.glob(vid_img_loc+"flow_y_*.jpg")
    flow_y.sort()

    print(len(images), len(flow_x), len(flow_y))
    assert len(images) == len(flow_x)
    assert len(flow_x) == len(flow_y)

    total_batches = int(len(images)/batch_size)

    for i in range(total_batches):
        m = i*batch_size
        n = (i+1)*batch_size
        if n > len(images): break
        images_rgb = images[m:n]
        flow_x_batch = flow_x[m:n]
        flow_y_batch = flow_y[m:n]

        rgb = read_vid_rgb(images_rgb, resize_len, crop_h, crop_w)
        rgb = np.expand_dims(rgb, axis=0)
        rgb = np.float32(rgb)

        for c in range(rgb.shape[4]):
            tmp = np.float32(rgb[:,:,:,:,c])
            rgb[:,:,:,:,c] = ((tmp - np.min(tmp)) / (np.max(tmp) - np.min(tmp)) - 0.5) * 2

        print(rgb.shape)


        flow = read_vid_flow(flow_x_batch, flow_y_batch, resize_len, crop_h, crop_w)
        flow = np.expand_dims(flow, axis=0)
        flow = np.float32(flow)

        for c in range(flow.shape[4]):
            tmp = np.float32(flow[:,:,:,:,c])
            flow[:,:,:,:,c] = ((tmp - np.min(tmp)) / (np.max(tmp) - np.min(tmp)) - 0.5) * 2

        print("start_frame: {}, end_frame: {}, rgb_shape: {}, flow_shape: {}".format(m, n, rgb.shape, flow.shape))
        #assert rgb.shape == flow.shape
        yield m, n, rgb, flow


def read_vid_flow(flow_x, flow_y, resize_len, crop_h, crop_w):
    """ read a list of flow images, combine x & y files and run it
    """
    vid = []
    for i, j in zip(flow_x, flow_y):
        imgx = cv2.imread(i, cv2.IMREAD_GRAYSCALE)
        imgy = cv2.imread(j, cv2.IMREAD_GRAYSCALE)

        h, w = imgx.shape[:2]
        resize_ratio = float(resize_len) / min(h, w)
        imgx = cv2.resize(imgx, None, fx=resize_ratio, fy=resize_ratio)
        imgy = cv2.resize(imgy, None, fx=resize_ratio, fy=resize_ratio)

        # crop center
        h, w = imgx.shape[:2]
        center_y = int((h - crop_h)/2)
        center_x = int((w - crop_w)/2)
        #print(center_y, cxenter_x, h, w, resize_ratio)
        imgx = imgx[center_y:center_y+crop_h, center_x:center_x+crop_w][:, :, np.newaxis]
        imgy = imgy[center_y:center_y+crop_h, center_x:center_x+crop_w][:, :, np.newaxis]
        img = np.concatenate((imgx, imgy), axis=2)
        vid.append(img[np.newaxis, :, :, :])
    return np.concatenate(vid)
