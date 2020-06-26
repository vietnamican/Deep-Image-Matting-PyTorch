import math
import argparse
import os

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


def process_test(name):
    img = cv.imread(out_path_test_composition + name)
    alpha = cv.imread(a_path_test_composition + name, 0)
    fg = cv.imread(fg_path_test_composition + name)
    bg = cv.imread(bg_path_test_composition + name)
    trimap = cv.imread(trimap_path_test_composition + name, 0)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    fg = cv.cvtColor(fg, cv.COLOR_BGR2RGB)
    bg = cv.cvtColor(bg, cv.COLOR_BGR2RGB)
    return img, alpha, fg, bg, trimap

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default='checkpoint.txt')
    parser.add_argument('--checkpoint', type=str, default='BEST_checkkpoint.tar')
    parser.add_argument('--output-folder', type=str)
    parser.add_argument('--device', type=str)
    args = parser.parse_args()
    ensure_folder('images' )
    ensure_folder('images/test' )
    ensure_folder('images/test/out' )
    ensure_folder('images/test/out/' + args.output_folder )
    f = open(args.file, "w")

    checkpoint = args.checkpoint
    if args.device == 'cpu':
        checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)
    else:
        checkpoint = torch.load(checkpoint)
    model = checkpoint['model']
    model = model.module.to(args.device)
    model.eval()

    transformer = data_transforms['valid']

    names = gen_test_names()

    mse_losses = AverageMeter()
    sad_losses = AverageMeter()
    gradient_losses = AverageMeter()
    connectivity_losses = AverageMeter()

    logger = get_logger()
    i = 0
    for name in tqdm(names):
        print(name)
        # fcount = int(name.split('.')[0].split('_')[0])
        # bcount = int(name.split('.')[0].split('_')[1])
        # im_name = fg_test_files[fcount]
        im_name = name
        print(out_path_test_composition + im_name)
        img = cv.imread(out_path_test_composition + im_name)
        trimap = cv.imread(trimap_path_test_composition + im_name, 0)
        h, w = img.shape[:2]

        x = torch.zeros((1, 4, h, w), dtype=torch.float)
        img = img[..., ::-1]  # RGB
        img = transforms.ToPILImage()(img)  # [3, 320, 320]
        img = transformer(img)  # [3, 320, 320]
        x[0:, 0:3, :, :] = img
        x[0:, 3, :, :] = torch.from_numpy(trimap.copy() / 255.)

        # Move to GPU, if available
        x = x.type(torch.FloatTensor).to(args.device)  # [1, 4, 320, 320]

        with torch.no_grad():
            pred = model(x)  # [1, 4, 320, 320]

        pred = pred.cpu().numpy()
        pred = pred.reshape((h, w))  # [320, 320]

        pred[new_trimap == 0] = 0.0
        pred[new_trimap == 255] = 1.0

        out = (pred.copy() * 255).astype(np.uint8)

        filename = os.path.join(args.output_folder, im_name)
        cv.imwrite(filename, out)
        print('wrote {}.'.format(filename))