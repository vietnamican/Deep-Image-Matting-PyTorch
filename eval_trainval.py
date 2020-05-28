import os

import cv2 as cv
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

from config import device
from data_gen import data_transforms
from utils import ensure_folder, compute_mse, compute_sad, draw_str


IMG_FOLDER = 'alphamatting/input_lowres'
ALPHA_FOLDER = 'alphamatting/gt_lowres'
TRIMAP_FOLDERS = ['alphamatting/trimap_lowres/Trimap1', 'alphamatting/trimap_lowres/Trimap2']
OUTPUT_FOLDERS = ['alphamatting/output_lowres_new26/Trimap1', 'alphamatting/output_lowres_new26/Trimap2', 'images/alphamatting/output_lowres_new26/Trimap3', ]

if __name__ == '__main__':
    # checkpoint = 'BEST_checkpoint.tar'
    # checkpoint = torch.load(checkpoint)
    # model = checkpoint['model'].module
    # model = model.to(device)
    # model.eval()
    checkpoint = 'checkpoints_1/checkpoint_26_0.05999396165859872.tar'
    checkpoint = torch.load(checkpoint)
    model = checkpoint['model']
    model = model.to(device)
    model.eval()
    # checkpoint = 'checkpoint_0007_0.0650.tar'
    # checkpoint = torch.load(checkpoint)
    # model = checkpoint['model']
    # model = model.to(device)
    # model.eval()

    transformer = data_transforms['valid']

    ensure_folder('images')
    ensure_folder('images/alphamatting')
    ensure_folder(OUTPUT_FOLDERS[0])
    ensure_folder(OUTPUT_FOLDERS[1])
    # ensure_folder(OUTPUT_FOLDERS[2])

    files = [f for f in os.listdir(IMG_FOLDER) if f.endswith('.png')]

    for file in tqdm(files):
        filename = os.path.join(IMG_FOLDER, file)
        img = cv.imread(filename)
        filename = os.path.join(ALPHA_FOLDER, file)
        # print(filename)
        alpha = cv.imread(filename, 0)
        alpha = alpha / 255
        print(img.shape)
        h, w = img.shape[:2]

        x = torch.zeros((1, 4, h, w), dtype=torch.float)
        image = img[..., ::-1]  # RGB
        image = transforms.ToPILImage()(image)
        image = transformer(image)
        x[0:, 0:3, :, :] = image

        for i in range(2):
            filename = os.path.join(TRIMAP_FOLDERS[i], file)
            print('reading {}...'.format(filename))
            trimap = cv.imread(filename, 0)
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

            # Calculate loss
            # loss = criterion(alpha_out, alpha_label)
            # print(pred.shape)
            # print(alpha.shape)
            mse_loss = compute_mse(pred, alpha, trimap)
            sad_loss = compute_sad(pred, alpha)
            str_msg = 'sad: %.4f, mse: %.4f' % (sad_loss, mse_loss)
            print(str_msg)

            out = (pred.copy() * 255).astype(np.uint8)

            draw_str(out, (10, 20), str_msg)
            filename = os.path.join(OUTPUT_FOLDERS[i], file)
            cv.imwrite(filename, out)
            print('wrote {}.'.format(filename))
