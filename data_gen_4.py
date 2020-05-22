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
from config import im_size, unknown_code, fg_path, bg_path, a_path, num_valid, valid_ratio
from utils import safe_crop, parse_args, maybe_random_interp

global args
args = parse_args()

num_fgs = 431
num_bgs_per_fg = 100
num_bgs = num_fgs * num_bgs_per_fg
split_ratio = 0.2

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.ColorJitter(brightness=0.125, contrast=0.125, saturation=0.125),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'valid': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

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

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
with open('Combined_Dataset/Training_set/training_fg_names.txt') as f:
    fg_files = f.read().splitlines()
with open('Combined_Dataset/Training_set/training_bg_names.txt') as f:
    bg_files = f.read().splitlines()
with open('Combined_Dataset/Test_set/test_fg_names.txt') as f:
    fg_test_files = f.read().splitlines()
with open('Combined_Dataset/Test_set/test_bg_names.txt') as f:
    bg_test_files = f.read().splitlines()


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


def gen_trimap(alpha):
    k_size = random.choice(range(1, 5))
    iterations = np.random.randint(1, 20)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (k_size, k_size))
    dilated = cv.dilate(alpha, kernel, iterations)
    eroded = cv.erode(alpha, kernel, iterations)
    trimap = np.zeros(alpha.shape)
    trimap.fill(128)
    trimap[eroded >= 255] = 255
    trimap[dilated <= 0] = 0
    return trimap


# Randomly crop (image, trimap) pairs centered on pixels in the unknown regions.
def random_choice(trimap, crop_size=(320, 320)):
    crop_height, crop_width = crop_size
    y_indices, x_indices = np.where(trimap == unknown_code)
    # print(y_indices)
    # print(x_indices)
    num_unknowns = len(y_indices)
    x, y = 0, 0
    if num_unknowns > 0:
        ix = np.random.choice(range(num_unknowns))
        center_x = x_indices[ix]
        center_y = y_indices[ix]
        x = max(0, center_x - int(crop_width / 2))
        y = max(0, center_y - int(crop_height / 2))
    return x, y

class DIMDataset(Dataset):
    def __init__(self, split):
        self.split = split

        names_train, names_valid = split_name()
        if self.split == "train":
            self.fgs = names_train
        else:
            self.fgs = names_valid

        self.fg_num_unique = len(self.fgs)
        self.fgs = np.repeat(self.fgs, args.batch_size * 8)
        print(len(self.fgs))
        self.fg_num = len(self.fgs)
        
        self.transformer = data_transforms[split]

        self.current_index = -1
        self.current_fg = None
        self.current_alpha = None
        self.is_resize = False

    def __getitem__(self, i):
        fcount = self.fgs[i]

        if i % args.batch_size == 0:
            self.current_index = fcount
            alpha = get_raw("a", fcount)
            alpha = np.reshape(alpha, (alpha.shape[0], alpha.shape[1]))
            fg = get_raw("fg", fcount)
            if args.data_augumentation:
                fg, alpha = self._composite_fg(alpha, fg, i)
            self.current_fg = fg
            self.current_alpha = alpha
            self.is_resize = True if np.random.rand() < 0.25 else False
        else:
            fg = self.current_fg
            alpha = self.current_alpha
        
        bcount = np.random.randint(num_bgs)
        img, _, _, bg = process(fcount, bcount)

        if self.is_resize:
            interpolation = maybe_random_interp(cv.INTER_NEAREST)
            img = cv.resize(img, (640, 640), interpolation=interpolation)
            # fg = cv.resize(fg, (640, 640), interpolation=interpolation)
            alpha = cv.resize(alpha, (640, 640), interpolation=interpolation)
            # bg = cv.resize(bg, (640, 640), interpolation=interpolation)

        # crop size 320:640:480 = 1:1:1
        different_sizes = [(320, 320), (480, 480), (640, 640)]
        crop_size = random.choice(different_sizes)

        trimap = gen_trimap(alpha)

        x, y = random_choice(trimap, crop_size)
        img = safe_crop(img, x, y, crop_size)
        alpha = safe_crop(alpha, x, y, crop_size)

        trimap = gen_trimap(alpha)

        # Flip array left to right randomly (prob=1:1)
        if np.random.random_sample() > 0.5:
            img = np.fliplr(img)
            trimap = np.fliplr(trimap)
            alpha = np.fliplr(alpha)

        x = torch.zeros((4, im_size, im_size), dtype=torch.float)
        img = img[..., ::-1]  # RGB
        img = transforms.ToPILImage()(img)
        img = self.transformer(img)
        x[0:3, :, :] = img
        x[3, :, :] = torch.from_numpy(trimap.copy() / 255.)

        y = np.empty((2, im_size, im_size), dtype=np.float32)
        y[0, :, :] = alpha / 255.
        mask = np.equal(trimap, 128).astype(np.float32)
        y[1, :, :] = mask

        return x, y

    def __len__(self):
        return len(self.fgs)

    def _composite_fg(self, alpha, fg, idx):
        if np.random.rand() < 0.5:
            idx2 = np.random.randint(self.fg_num_unique) + idx
            alpha2 = get_raw("a", idx2 % self.fg_num_unique)
            alpha2 = np.reshape(alpha2, (alpha2.shape[0], alpha2.shape[1]))
            fg2 = get_raw("fg", idx2 % self.fg_num_unique)
            h, w = alpha.shape
            interpolation = maybe_random_interp(cv.INTER_NEAREST)
            fg2 = cv.resize(fg2, (w, h), interpolation=interpolation)
            alpha2 = cv.resize(alpha2, (w, h), interpolation=interpolation)
            alpha_tmp = 1 - (1 - alpha / 255.0) * (1 - alpha2 / 255.0)
            if  np.any(alpha_tmp < 1):
                fg, alpha, _, _ = composite4(fg, fg2, alpha, w, h)
                alpha = alpha_tmp * 255.0
        return fg, alpha

def split_name():
    names = list(range(num_fgs))
    np.random.shuffle(names)
    split_index = math.ceil(num_fgs - num_fgs * valid_ratio)
    names_train = np.copy(names)
    names_valid = np.copy(names)
    return names_train, names_valid

if __name__ == "__main__":
    names_train, names_valid = split_name()
    print(names_train)
    names_train, names_valid = split_name()
    print(names_train)
