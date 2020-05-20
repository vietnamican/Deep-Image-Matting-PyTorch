import math
import os
import random

import cv2 as cv
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import tensorflow as tf

import tfrecord_creator	
from utils import safe_crop, parse_args, maybe_random_interp

with open('Combined_Dataset/Training_set/training_fg_names.txt') as f:
    fg_files = f.read().splitlines()
with open('Combined_Dataset/Training_set/training_bg_names.txt') as f:
    bg_files = f.read().splitlines()
with open('Combined_Dataset/Test_set/test_fg_names.txt') as f:
    fg_test_files = f.read().splitlines()
with open('Combined_Dataset/Test_set/test_bg_names.txt') as f:
    bg_test_files = f.read().splitlines()

def return_raw_image(dataset):
    dataset_raw = []
    for image_features in dataset:
        image_raw = image_features['image'].numpy()
        image = tf.image.decode_jpeg(image_raw)
        dataset_raw.append(image)
        
    return dataset_raw

fg_dataset = tfrecord_creator.read("fg", "./data/tfrecord/")
bg_dataset = tfrecord_creator.read("bg", "./data/tfrecord/")
a_dataset  = tfrecord_creator.read("a",  "./data/tfrecord/")
fg_dataset = list(fg_dataset)
bg_dataset = list(bg_dataset)
a_dataset  = list(a_dataset)
# fg_raw = return_raw_image(fg_dataset)
# bg_raw = return_raw_image(bg_dataset)
# a_raw  = return_raw_image(a_dataset)



def get_raw(type_of_dataset, count):
    if type_of_dataset == 'fg':
        temp = fg_dataset[count]['image']
        channels=3
    elif type_of_dataset == 'bg':
        temp = bg_dataset[count]['image']
        channels=3
    else :
        temp = a_dataset[count]['image']
        channels=0
    temp = tf.image.decode_jpeg(temp, channels=channels)
    temp = np.asarray(temp)
    return temp

def get_alpha(name):
    fg_i = int(name.split("_")[0])
    name = fg_files[fg_i]
    filename = os.path.join('data/mask', name)
    alpha = cv.imread(filename, 0)
    return alpha


def get_alpha_test(name):
    fg_i = int(name.split("_")[0])
    name = fg_test_files[fg_i]
    filename = os.path.join('data/mask_test', name)
    alpha = cv.imread(filename, 0)
    return alpha


def composite4(fg, bg, a, w, h):
    fg = np.array(fg, np.float32)
    bg_h, bg_w = bg.shape[:2]
    x = 0
    if bg_w > w:
        x = np.random.randint(0, bg_w - w)
    y = 0
    if bg_h > h:
        y = np.random.randint(0, bg_h - h)
    if bg.ndim == 2:
        bg = np.reshape(bg, (h,w,1))
    bg = np.array(bg[y:y + h, x:x + w], np.float32)
    bg = np.reshape(bg, (h,w,-1))
    fg = np.reshape(fg, (h,w,-1))
    alpha = np.zeros((h, w, 1), np.float32)
    alpha[:, :, 0] = a / 255.
    im = alpha * fg + (1 - alpha) * bg
    im = im.astype(np.uint8)
    return im, a, fg, bg


def process(fcount, bcount):
    im = get_raw("fg", fcount)
    a = get_raw("a", fcount)
    a = np.reshape(a, (a.shape[0], a.shape[1]))
    h, w = im.shape[:2]
    bg = get_raw("bg", bcount)
    bh, bw = bg.shape[:2]
    wratio = w / bw
    hratio = h / bh
    ratio = wratio if wratio > hratio else hratio
    if ratio > 1:
        bg = cv.resize(src=bg, dsize=(math.ceil(bw * ratio), math.ceil(bh * ratio)), interpolation=cv.INTER_CUBIC)

    return composite4(im, bg, a, w, h)

def _composite_fg(img, alpha, fg, bg, idx):
        idx2 = 3
        img2, alpha2, fg2, bg2 = process(idx2, 1)
        h, w = alpha.shape
        fg2 = cv.resize(fg2, (w, h), interpolation=maybe_random_interp(cv.INTER_NEAREST))
        alpha2 = cv.resize(alpha2, (w, h), interpolation=maybe_random_interp(cv.INTER_NEAREST))
        cv.imshow("a", fg[:,:,::-1])
        cv.waitKey(0)
        cv.destroyAllWindows()
        cv.imshow("a", fg2[:,:,::-1])
        cv.waitKey(0)
        cv.destroyAllWindows()
        alpha_tmp = 1 - (1 - alpha) * (1 - alpha2)
        if  np.any(alpha_tmp < 1):
            fg = fg.astype(np.float32) * alpha[:,:,None] + fg2.astype(np.float32) * (1 - alpha[:,:,None])
            # The overlap of two 50% transparency should be 25%
            # alpha = alpha_tmp
            fg = fg.astype(np.uint8)
        cv.imshow("a", fg[:,:,::-1])
        cv.waitKey(0)
        cv.destroyAllWindows()
        img, alpha, fg, bg = composite4(fg, bg, alpha, w, h)
        if np.random.rand() < 0.25:
            fg = cv.resize(fg, (640, 640), interpolation=maybe_random_interp(cv.INTER_NEAREST))
            alpha = cv.resize(alpha, (640, 640), interpolation=maybe_random_interp(cv.INTER_NEAREST))
        return img, alpha

if __name__ == "__main__":
    img, alpha, fg, bg = process(2, 18)
    cv.imshow("a", fg[:,:,::-1])
    cv.waitKey(0)
    cv.destroyAllWindows()
    img, alpha = _composite_fg(img, alpha, fg, bg, 1)
    cv.imshow("a", img[:,:,::-1])
    cv.waitKey(0)
    cv.destroyAllWindows()