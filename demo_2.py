import os
import argparse

import cv2 as cv
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='BEST_checkkpoint.tar')
    parser.add_argument('--image', type=str)
    parser.add_argument('--trimap', type=str)
    parser.add_argument('--result', type=str)
    parser.add_argument('--rgb-result', type=str)
    parser.add_argument('--device', type=str)

    args = parser.parse_args()
    checkpoint = args.checkpoint
    if args.device == 'cpu':
        checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)
    else:
        checkpoint = torch.load(checkpoint)
    model = checkpoint['model']
    model = model.module.to(args.device)
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
    x = x.type(torch.FloatTensor).to(args.device)

    with torch.no_grad():
        pred = model(x)

    pred = pred.cpu().numpy()
    pred = pred.reshape((h, w))

    pred[trimap == 0] = 0.0
    pred[trimap == 255] = 1.0

    out = (pred.copy() * 255).astype(np.uint8)

    image = img.copy()
    image = np.transpose(image, (2,0,1))
    rgb_image = image * pred
    rgb_image = np.transpose(rgb_image,(1,2,0))

    cv.imwrite(args.rgb_result, rgb_image)
    cv.imwrite(args.result, out)
