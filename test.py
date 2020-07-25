import math
import argparse
import gc

import cv2 as cv
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

# from config import device, fg_path_test, a_path_test, bg_path_test
from data_gen import data_transforms, fg_test_files, bg_test_files
from utils import compute_mse, compute_sad, compute_gradient_loss, compute_connectivity_error, AverageMeter, get_logger, \
    draw_str, ensure_folder, create_patches, assemble_patches, patch_dims

MAX_H = 1000
MAX_W = 1000


def gen_test_names(args):
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


def process_test(im_name, bg_name, trimap, args):
    gc.collect()
    print(args.fg_path + im_name)
    im = cv.imread(args.fg_path + im_name)
    print(args.a_path + im_name)
    a = cv.imread(args.a_path + im_name, 0)
    h, w = im.shape[:2]
    if h > MAX_H and w > MAX_W:
        crop_x = np.random.randint(0, h - MAX_H)
        crop_y = np.random.randint(0, w - MAX_W)
        im = im[crop_x:crop_x + MAX_H, crop_y:crop_y + MAX_W]
        a = a[crop_x:crop_x + MAX_H, crop_y:crop_y + MAX_W]
        trimap = trimap[crop_x:crop_x + MAX_H, crop_y:crop_y + MAX_W]
        h = MAX_H
        w = MAX_W
    print(args.bg_path + bg_name)
    bg = cv.imread(args.bg_path + bg_name)
    bh, bw = bg.shape[:2]
    wratio = w / bw
    hratio = h / bh
    ratio = wratio if wratio > hratio else hratio
    if ratio > 1:
        bg = cv.resize(src=bg, dsize=(math.ceil(bw * ratio), math.ceil(bh * ratio)), interpolation=cv.INTER_CUBIC)

    return composite4_test(im, bg, a, w, h, trimap, args)


def composite4_test(fg, bg, a, w, h, trimap, args):
    gc.collect()
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
    print(x, y, h, w, bg_h, bg_w)
    print(trimap.shape)
    new_a[y:y + h, x:x + w] = a
    new_trimap = np.zeros((bg_h, bg_w), np.uint8)
    print(new_trimap.shape)
    new_trimap[y:y + h, x:x + w] = trimap
    #     cv.imwrite('images/test/new/' + trimap_name, new_trimap)
    new_im = bg.copy()
    new_im[y:y + h, x:x + w] = im
    # cv.imwrite('images/test/new_im/'+trimap_name,new_im)
    return new_im, new_a, fg, bg, new_trimap


# class Args():
#     def __init__(self):
#         # self.checkpoint = "checkpoints/BEST_checkpoint.tar"
#         self.checkpoint = "checkpoints/checkpoint_14_0.017443798415107854.tar"
#         self.device = "cpu"
#         self.file="checkpoints/old.txt"
#         self.output_folder = "older"
#         self.fg_path = "./data/fg_test/"
#         self.bg_path = "./data/bg_test/"
#         self.a_path = "./data/mask_test/"
#         self.trimap_path = "./data/trimap_test/"

# args = Args()

if __name__ == '__main__':
    # args = Args()
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default='checkpoint.txt')
    parser.add_argument('--checkpoint', type=str, default='BEST_checkkpoint.tar')
    parser.add_argument('--output-folder', type=str)
    parser.add_argument('--device', type=str)
    parser.add_argument('--bg-path', type=str)
    parser.add_argument('--fg-path', type=str)
    parser.add_argument('--a-path', type=str)
    parser.add_argument('--trimap-path', type=str)
    args = parser.parse_args()
    ensure_folder('images')
    ensure_folder('images/test')
    ensure_folder('images/test/out')
    ensure_folder('images/test/out/' + args.output_folder)
    # ensure_folder('images/test/out/' + args.output_folder + '/Trimap1')
    # ensure_folder('images/test/out/' + args.output_folder + '/Trimap2')
    # ensure_folder('images/test/out/' + args.output_folder + '/Trimap3')
    f = open(args.file, "w")

    checkpoint = args.checkpoint
    if args.device == 'cpu':
        checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)
    else:
        checkpoint = torch.load(checkpoint)
    model = checkpoint['model'].module
    del checkpoint
    model.to(args.device)
    model.eval()

    transformer = data_transforms['valid']

    names = gen_test_names(args)

    mse_losses = AverageMeter()
    sad_losses = AverageMeter()

    logger = get_logger()
    i = 0
    # j = 0
    for name in tqdm(names):
        # gc.collect()
        # j += 1
        # if j == 3:
        #     break
        fcount = int(name.split('.')[0].split('_')[0])
        bcount = int(name.split('.')[0].split('_')[1])
        im_name = fg_test_files[fcount]
        # print(im_name)
        bg_name = bg_test_files[bcount]
        trimap_name = im_name.split('.')[0] + '_' + str(i) + '.png'
        # print('trimap_name: ' + str(trimap_name))

        trimap = cv.imread(args.trimap_path + trimap_name, 0)
        # print('trimap: ' + str(trimap))

        i += 1
        if i == 20:
            i = 0

        img, alpha, fg, bg, new_trimap = process_test(im_name, bg_name, trimap, args)
        h, w = img.shape[:2]
        # mytrimap = gen_trimap(alpha)
        # cv.imwrite('images/test/new_im/'+trimap_name,mytrimap)

        x = torch.zeros((1, 4, h, w), dtype=torch.float)
        img = img[..., ::-1]  # RGB
        img = transforms.ToPILImage()(img)  # [3, 320, 320]
        img = transformer(img)  # [3, 320, 320]
        x[0:, 0:3, :, :] = img
        x[0:, 3, :, :] = torch.from_numpy(new_trimap.copy() / 255.)

        # Move to GPU, if available
        x = x.type(torch.FloatTensor).to(args.device)  # [1, 4, 320, 320]
        alpha = alpha / 255.

        #         if x.shape[2] > 5800 and x.shape[3] > 5800:
        #             PATCH_SIZE = 320
        #             # print(x.shape)
        #             patches = create_patches(x, PATCH_SIZE)
        #             patches_count = np.product(
        #                 patch_dims(mat_size=new_trimap.shape, patch_size=PATCH_SIZE)
        #             )
        #             patches_predictions = np.zeros(shape=(patches_count, PATCH_SIZE, PATCH_SIZE))
        #             patches = torch.Tensor(patches).cuda()
        #             for i in range(patches.shape[0]):
        #                 print("Predicting patches {}/{}".format(i + 1, patches_count))
        #                 with torch.no_grad():
        #                     print(patches[i, None, :, :, :].shape)
        #                     patch_prediction = model(patches[i, None, :, :, :])
        #                     print("patch_prediction", patch_prediction.shape)
        #                 patches_predictions[i] = np.reshape(patch_prediction.cpu(), (PATCH_SIZE, PATCH_SIZE)) * 255.
        #             print(patches_predictions.shape)
        #             pred = assemble_patches(patches_predictions, new_trimap.shape, PATCH_SIZE)
        #             pred = pred[:x.shape[2], :x.shape[3]]
        #             pred = torch.Tensor(pred)
        #         else:
        with torch.no_grad():
            pred = model(x)  # [1, 4, 320, 320]
        torch.cuda.empty_cache()
        pred = pred.cpu().numpy()
        pred = pred.reshape((h, w))  # [320, 320]

        pred[new_trimap == 0] = 0.0
        pred[new_trimap == 255] = 1.0

        # Calculate loss
        # loss = criterion(alpha_out, alpha_label)
        mse_loss = compute_mse(pred, alpha, new_trimap)
        sad_loss = compute_sad(pred, alpha)

        # Keep track of metrics
        mse_losses.update(mse_loss.item())
        sad_losses.update(sad_loss.item())
        print("sad:{} mse:{}".format(sad_loss.item(), mse_loss.item()))
        print("sad_avg:{} mse_avg:{}".format(sad_losses.avg, mse_losses.avg))
        f.write("sad:{} mse:{}".format(sad_loss.item(), mse_loss.item()) + "\n")

        pred = (pred.copy() * 255).astype(np.uint8)
        draw_str(pred, (10, 20), "sad:{} mse:{}".format(sad_loss.item(), mse_loss.item()))
        cv.imwrite('images/test/out/' + args.output_folder + '/' + trimap_name, pred)

    print("sad_avg:{} mse_avg:{}".format(sad_losses.avg, mse_losses.avg))
    f.write("sad_avg:{} mse_avg:{}".format(sad_losses.avg, mse_losses.avg) + "\n")
