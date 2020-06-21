import math
import argparse

import cv2 as cv
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

from config import device, fg_path_test, a_path_test, bg_path_test, out_path_test, fg_path_test_composition, a_path_test_composition, bg_path_test_composition, out_path_test_composition, trimap_path_test_composition
from data_gen import data_transforms, fg_test_files, bg_test_files
from utils import compute_mse, compute_sad, compute_gradient_loss, compute_connectivity_error, AverageMeter, get_logger, draw_str, ensure_folder


def gen_test_names():
    num_fgs = 50
    num_bgs = 1000
    num_bgs_per_fg = 20

    names = []
    bcount = 0
    for fcount in range(num_fgs):
        for i in range(num_bgs_per_fg):
            names.append(str(fcount) + '_' + str(bcount) + '.png')
            bcount += 1

    return names



def process_test(im_name, bg_name, trimap):
    # print(bg_path_test + bg_name)
    im = cv.imread(fg_path_test + im_name)
    a = cv.imread(a_path_test + im_name, 0)
    h, w = im.shape[:2]
    bg = cv.imread(bg_path_test + bg_name)
    bh, bw = bg.shape[:2]
    wratio = w / bw
    hratio = h / bh
    ratio = wratio if wratio > hratio else hratio
    if ratio > 1:
        bg = cv.resize(src=bg, dsize=(math.ceil(bw * ratio), math.ceil(bh * ratio)), interpolation=cv.INTER_CUBIC)

    return composite4_test(im, bg, a, w, h, trimap)


# def composite4_test(fg, bg, a, w, h):
#     fg = np.array(fg, np.float32)
#     bg_h, bg_w = bg.shape[:2]
#     x = max(0, int((bg_w - w)/2))
#     y = max(0, int((bg_h - h)/2))
#     bg = np.array(bg[y:y + h, x:x + w], np.float32)
#     alpha = np.zeros((h, w, 1), np.float32)
#     alpha[:, :, 0] = a / 255.
#     im = alpha * fg + (1 - alpha) * bg
#     im = im.astype(np.uint8)
#     print('im.shape: ' + str(im.shape))
#     print('a.shape: ' + str(a.shape))
#     print('fg.shape: ' + str(fg.shape))
#     print('bg.shape: ' + str(bg.shape))
#     return im, a, fg, bg


def composite4_test(fg, bg, a, w, h, trimap):
    fg = np.array(fg, np.float32)
    bg_h, bg_w = bg.shape[:2]
    x = max(0, int((bg_w - w) / 2))
    y = max(0, int((bg_h - h) / 2))
    crop = np.array(bg[y:y + h, x:x + w], np.float32)
    alpha = np.zeros((h, w, 1), np.float32)
    alpha[:, :, 0] = a / 255.
    # trimaps = np.zeros((h, w, 1), np.float32)
    # trimaps[:,:,0]=trimap/255.

    im = alpha * fg + (1 - alpha) * crop
    im = im.astype(np.uint8)

    new_a = np.zeros((bg_h, bg_w), np.uint8)
    new_a[y:y + h, x:x + w] = a
    new_trimap = np.zeros((bg_h, bg_w), np.uint8)
    new_trimap[y:y + h, x:x + w] = trimap
    # cv.imwrite('images/test/new/' + trimap_name, new_trimap)
    new_im = bg.copy()
    new_im[y:y + h, x:x + w] = im
    # cv.imwrite('images/test/new_im/'+trimap_name,new_im)
    return new_im, new_a, fg, bg, new_trimap

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--output-folder', type=str)
    # args = parser.parse_args()
    ensure_folder(out_path_test_composition)
    ensure_folder(a_path_test_composition)
    ensure_folder(fg_path_test_composition)
    ensure_folder(bg_path_test_composition)
    ensure_folder(trimap_path_test_composition)

    names = gen_test_names()

    logger = get_logger()
    i = 0
    for name in tqdm(names):
        fcount = int(name.split('.')[0].split('_')[0])
        bcount = int(name.split('.')[0].split('_')[1])
        im_name = fg_test_files[fcount]
        # print(im_name)
        bg_name = bg_test_files[bcount]
        trimap_name = im_name.split('.')[0] + '_' + str(i) + '.png'
        # print('trimap_name: ' + str(trimap_name))

        trimap = cv.imread('data/Combined_Dataset/Test_set/Adobe-licensed images/trimaps/' + trimap_name, 0) 
        # print('trimap: ' + str(trimap))

        i += 1
        if i == 20:
            i = 0

        img, alpha, fg, bg, trimap = process_test(im_name, bg_name, trimap)

        cv.imwrite(out_path_test_composition + name, img)
        cv.imwrite(a_path_test_composition + name, alpha)
        cv.imwrite(fg_path_test_composition + name, fg)
        cv.imwrite(bg_path_test_composition + name, bg)
        cv.imwrite(trimap_path_test_composition + name, trimap)

        
