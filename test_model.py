import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm
import os 
import cv2 as cv
import numpy as np

import models
import models_v16
from config import device

# old_model = models.DIMModel()
# new_model = models_v16.DIMModel()

from config import device
from data_gen import data_transforms
from utils import ensure_folder, compute_mse, compute_sad, draw_str


IMG_FOLDER = 'alphamatting/input_lowres'
ALPHA_FOLDER = 'alphamatting/gt_lowres'
TRIMAP_FOLDERS = ['alphamatting/trimap_lowres/Trimap1', 'alphamatting/trimap_lowres/Trimap2']
OUTPUT_FOLDERS = ['alphamatting/output_lowres_older1/Trimap1', 'alphamatting/output_lowres_older1/Trimap2', 'images/alphamatting/output_lowres_older1/Trimap3', ]


def migrate(new_model):
    # print(new_model)
    checkpoint = 'BEST_checkpoint_older.tar'
    checkpoint = torch.load(checkpoint)
    old_model = checkpoint['model'].module
    # print(dict(old_model.up1.unpool.named_parameters()))
    # print(old_model)
    # print("old")
    # print(dict(old_model.up1.conv.cbr_unit[0].named_parameters()))
    # print("new")
    # print(dict(new_model.up1.conv.cbr_unit[0].named_parameters()))
    l1s = [
        old_model.down1.conv1.cbr_unit[0],
        old_model.down1.conv1.cbr_unit[1],
        old_model.down1.conv2.cbr_unit[0],
        old_model.down1.conv2.cbr_unit[1],
        old_model.down2.conv1.cbr_unit[0],
        old_model.down2.conv1.cbr_unit[1],
        old_model.down2.conv2.cbr_unit[0],
        old_model.down2.conv2.cbr_unit[1],
        old_model.down3.conv1.cbr_unit[0],
        old_model.down3.conv1.cbr_unit[1],
        old_model.down3.conv2.cbr_unit[0],
        old_model.down3.conv2.cbr_unit[1],
        old_model.down3.conv3.cbr_unit[0],
        old_model.down3.conv3.cbr_unit[1],
        old_model.down4.conv1.cbr_unit[0],
        old_model.down4.conv1.cbr_unit[1],
        old_model.down4.conv2.cbr_unit[0],
        old_model.down4.conv2.cbr_unit[1],
        old_model.down4.conv3.cbr_unit[0],
        old_model.down4.conv3.cbr_unit[1],
        old_model.down5.conv1.cbr_unit[0],
        old_model.down5.conv1.cbr_unit[1],
        old_model.down5.conv2.cbr_unit[0],
        old_model.down5.conv2.cbr_unit[1],
        old_model.down5.conv3.cbr_unit[0],
        old_model.down5.conv3.cbr_unit[1],
        old_model.up5.conv.cbr_unit[0],
        old_model.up5.conv.cbr_unit[1],
        old_model.up4.conv.cbr_unit[0],
        old_model.up4.conv.cbr_unit[1],
        old_model.up3.conv.cbr_unit[0],
        old_model.up3.conv.cbr_unit[1],
        old_model.up2.conv.cbr_unit[0],
        old_model.up2.conv.cbr_unit[1],
        old_model.up1.conv.cbr_unit[0],
        old_model.up1.conv.cbr_unit[1]
    ]

    l2s = [
        new_model.down1.conv1.cbr_unit[0],
        new_model.down1.conv1.cbr_unit[1],
        new_model.down1.conv2.cbr_unit[0],
        new_model.down1.conv2.cbr_unit[1],
        new_model.down2.conv1.cbr_unit[0],
        new_model.down2.conv1.cbr_unit[1],
        new_model.down2.conv2.cbr_unit[0],
        new_model.down2.conv2.cbr_unit[1],
        new_model.down3.conv1.cbr_unit[0],
        new_model.down3.conv1.cbr_unit[1],
        new_model.down3.conv2.cbr_unit[0],
        new_model.down3.conv2.cbr_unit[1],
        new_model.down3.conv3.cbr_unit[0],
        new_model.down3.conv3.cbr_unit[1],
        new_model.down4.conv1.cbr_unit[0],
        new_model.down4.conv1.cbr_unit[1],
        new_model.down4.conv2.cbr_unit[0],
        new_model.down4.conv2.cbr_unit[1],
        new_model.down4.conv3.cbr_unit[0],
        new_model.down4.conv3.cbr_unit[1],
        new_model.down5.conv1.cbr_unit[0],
        new_model.down5.conv1.cbr_unit[1],
        new_model.down5.conv2.cbr_unit[0],
        new_model.down5.conv2.cbr_unit[1],
        new_model.down5.conv3.cbr_unit[0],
        new_model.down5.conv3.cbr_unit[1],
        new_model.up5.conv.cbr_unit[0],
        new_model.up5.conv.cbr_unit[1],
        new_model.up4.conv.cbr_unit[0],
        new_model.up4.conv.cbr_unit[1],
        new_model.up3.conv.cbr_unit[0],
        new_model.up3.conv.cbr_unit[1],
        new_model.up2.conv.cbr_unit[0],
        new_model.up2.conv.cbr_unit[1],
        new_model.up1.conv.cbr_unit[0],
        new_model.up1.conv.cbr_unit[1]
    ]

    for l1, l2 in zip(l1s, l2s):
        if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
            if l1.weight.size() == l2.weight.size() and l1.bias.size() == l2.bias.size():
                print("success")
                # l2.weight.data.copy_(l1.weight.data)
                l2.weight.data = l1.weight.data
                # l2.bias.data.copy_(l1.bias.data)
                l2.bias.data = l1.bias.data
        elif isinstance(l1, nn.BatchNorm2d) and isinstance(l2, nn.BatchNorm2d):
            if l1.weight.size() == l2.weight.size() and l1.bias.size() == l2.bias.size():
                print("success")
                # l2.weight.data.copy_(l1.weight.data)
                l2.weight.data = l1.weight.data
                # l2.bias.data.copy_(l1.bias.data)
                l2.bias.data = l1.bias.data
                l2.running_mean.data = l1.running_mean.data
                l2.running_var.data = l1.running_var.data
    del checkpoint
    # print("old")
    # print(dict(old_model.up1.conv.cbr_unit[0].named_parameters()))
    # print("new")
    # print(dict(new_model.up1.conv.cbr_unit[0].named_parameters()))
    # new_model.load_state_dict(old_model.state_dict())


if __name__ == "__main__":
    model = models.DIMModel()
    migrate(model)
    # print(dict(model.up1.conv.cbr_unit[0].named_parameters()))

    model = model.to(device)
    model.eval()

    # checkpoint = 'BEST_checkpoint_older.tar'
    # checkpoint = torch.load(checkpoint)
    # old_model = checkpoint['model'].module
    # # print(old_model.state_dict())
    # print(old_model)

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