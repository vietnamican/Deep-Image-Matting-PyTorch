import os
import argparse

import cv2 as cv
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

from config import device
from data_gen import data_transforms
from utils import ensure_folder, parse_args


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='BEST_checkkpoint.tar')
    parser.add_argument('--image', type=str)
    parser.add_argument('--trimap', type=str)

    args = parser.parse_args()
    checkpoint = args.checkpoint
    if device == 'cpu':
        checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)
    else:
        checkpoint = torch.load(checkpoint)
    model = checkpoint['model']
    model = model.to(device)
    model.eval()

    transformer = data_transforms['valid']

    # files = [f for f in os.listdir(IMG_FOLDER) if f.endswith('.png')]

    # for file in tqdm(files):
    img = cv.imread(args.image)
    # print(img.shape)
    h, w = img.shape[:2]

    x = torch.zeros((1, 4, h, w), dtype=torch.float)
    image = img[..., ::-1]  # RGB
    image = transforms.ToPILImage()(image)
    image = transformer(image)
    x[0:, 0:3, :, :] = image

    trimap = cv.imread(args.trimap, 0)
    x[0:, 3, :, :] = torch.from_numpy(trimap.copy() / 255.)
    # print(torch.max(x[0:, 3, :, :]))
    # print(torch.min(x[0:, 3, :, :]))
    # print(torch.median(x[0:, 3, :, :]))

    # Move to GPU, if available
    x = x.type(torch.FloatTensor).to(device)

    with torch.no_grad():
        pred = model(x)

    pred = pred.cpu().numpy()
    pred = pred.reshape((h, w))

    pred[trimap == 0] = 0.0
    pred[trimap == 255] = 1.0

    out = (pred.copy() * 255).astype(np.uint8)

    image = img.copy()
    # print(pred.dtype)
    print(image.shape)
    image = np.transpose(image, (2,0,1))
    print(image.dtype)
    rgb_image = image * pred
    rgb_image = np.transpose(rgb_image,(1,2,0))
    # rgb_image = rgb_image[...,::-1]
    print(rgb_image.shape)
    cv.imwrite('rgb.png', rgb_image)

    cv.imwrite('result.png', out)
    print('wrote {}.'.format("result.png"))
