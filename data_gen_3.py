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
from config import im_size, unknown_code, fg_path, bg_path, a_path, num_valid
from utils import safe_crop, parse_args, maybe_random_interp

global args
args = parse_args()

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

        filename = '{}_names.txt'.format(split)
        with open(filename, 'r') as file:
            self.names = file.read().splitlines()

        fgs = [int(name.split('.')[0].split('_')[0]) for name in self.names]
        bgs = [int(name.split('.')[0].split('_')[1]) for name in self.names]
        random.shuffle(fgs)
        random.shuffle(bgs)
        self.fgs = fgs
        self.bgs = bgs
        self.fg_num = np.unique(self.fgs).shape[0]
        self.bg_num = np.unique(self.bgs).shape[0]
        print(self.fg_num)
        print(self.bg_num)

        self.transformer = data_transforms[split]

    def __getitem__(self, i):
        fcount = self.fgs[i]
        bcount = self.bgs[i]
        img, alpha, fg, bg = process(fcount, bcount)

        # crop size 320:640:480 = 1:1:1
        different_sizes = [(320, 320), (480, 480), (640, 640)]
        crop_size = random.choice(different_sizes)

        trimap = gen_trimap(alpha)

        if args.data_augumentation:
            img, alpha = self._composite_fg(img, alpha, fg, bg, i)

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

        if(i >= self.__len__() - 1):
            fgs = self.fgs
            bgs = self.bgs
            random.shuffle(fgs)
            random.shuffle(bgs)
            self.fgs = fgs
            self.bgs = bgs

        return x, y

    def __len__(self):
        return len(self.names)

    def _composite_fg(self, img, alpha, fg, bg, idx):
        if np.random.rand() < 0.5:
            idx2 = np.random.randint(self.fg_num) + idx
            img2, alpha2, fg2, bg2 = process(idx2 % self.fg_num, 0)
            h, w = alpha.shape
            fg2 = cv.resize(fg2, (w, h), interpolation=maybe_random_interp(cv.INTER_NEAREST))
            alpha2 = cv.resize(alpha2, (w, h), interpolation=maybe_random_interp(cv.INTER_NEAREST))
            alpha_tmp = 1 - (1 - alpha / 255.0) * (1 - alpha2 / 255.0)
            if  np.any(alpha_tmp < 1):
                img, alpha, _, _ = composite4(fg, fg2, alpha, w, h)
                alpha = alpha_tmp * 255.0
                img = img.astype(np.uint8)
            img, alpha, fg, bg = composite4(img, bg, alpha, w, h)
        if np.random.rand() < 0.25:
            img = cv.resize(img, (640, 640), interpolation=maybe_random_interp(cv.INTER_NEAREST))
            alpha = cv.resize(alpha, (640, 640), interpolation=maybe_random_interp(cv.INTER_NEAREST))
        return img, alpha

def gen_names():
    num_fgs = 431
    num_bgs = 43100
    num_bgs_per_fg = 100

    names = []
    bcount = 0
    for fcount in range(num_fgs):
        for i in range(num_bgs_per_fg):
            names.append(str(fcount) + '_' + str(bcount) + '.png')
            bcount += 1

    valid_names = random.sample(names, num_valid)
    train_names = [n for n in names if n not in valid_names]

    with open('valid_names.txt', 'w') as file:
        file.write('\n'.join(valid_names))

    with open('train_names.txt', 'w') as file:
        file.write('\n'.join(train_names))


if __name__ == "__main__":
    gen_names()
    img1, alpha1, fg1, bg1 = process(1, 0)
    h, w = alpha1.shape
    img2, alpha2, fg2, bg2 = process(3, 2)
    cv.waitKey(0)
    cv.destroyAllWindows()
    fg2 = cv.resize(fg2, (w, h), interpolation=maybe_random_interp(cv.INTER_NEAREST))
    alpha2 = cv.resize(alpha2, (w, h), interpolation=maybe_random_interp(cv.INTER_NEAREST))
    alpha_tmp = 1 - (1 - alpha1 / 255.0) * (1 - alpha2 / 255.0)
    cv.imshow("1", img1)
    cv.imshow("2", img2)