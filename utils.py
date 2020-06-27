import argparse
import logging
import os
import math
import random

import cv2 as cv
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Sampler
from skimage.measure import label
import scipy.ndimage
import scipy.ndimage.morphology
from torchvision import transforms

from config import im_size, epsilon, epsilon_sqr, device


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.
    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(epoch, epochs_since_improvement, model, optimizer, loss, is_best, logdir):
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'loss': loss,
             'model': model,
             'optimizer': optimizer,
             'torch_seed': torch.get_rng_state(),
             'torch_cuda_seed': torch.cuda.get_rng_state(),
             'np_seed': np.random.get_state(),
             'python_seed': random.getstate()}
    filename = logdir + '/checkpoint_' + str(epoch) + '_' + str(loss) + '.tar'
    # filename = 'checkpoint.tar'
    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, logdir + '/BEST_checkpoint.tar')

def save_checkpoint_2(epoch, epochs_since_improvement, model, optimizer, loss, is_best):
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'loss': loss,
             'model': model,
             'optimizer': optimizer}
    filename = 'checkpoints_2/checkpoint_' + str(epoch) + '_' + str(loss) + '.tar'
    # filename = 'checkpoint.tar'
    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, 'checkpoints_2/BEST_checkpoint.tar')

def save_checkpoint_4(epoch, epochs_since_improvement, model, optimizer, loss, is_best):
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'loss': loss,
             'model': model,
             'optimizer': optimizer}
    filename = 'checkpoints_4/checkpoint_' + str(epoch) + '_' + str(loss) + '.tar'
    # filename = 'checkpoint.tar'
    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, 'checkpoints_4/BEST_checkpoint.tar')

def save_checkpoint_5(epoch, epochs_since_improvement, model, optimizer, loss, is_best):
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'loss': loss,
             'model': model,
             'optimizer': optimizer}
    filename = 'checkpoints_5/checkpoint_' + str(epoch) + '_' + str(loss) + '.tar'
    # filename = 'checkpoint.tar'
    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, 'checkpoints_5/BEST_checkpoint.tar')

class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.
    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def get_learning_rate(optimizer):
    return optimizer.param_groups[0]['lr']


def accuracy(scores, targets, k=1):
    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)


def parse_args():
    parser = argparse.ArgumentParser(description='Train face network')
    # general
    parser.add_argument('--checkpointdir', type=str)
    parser.add_argument('--logdir', type=str)
    parser.add_argument('--random-interp', type=bool, default=True, help='randomly choose interpolation')
    parser.add_argument('--start-epoch', type=int, default=0, help='start epoch.')
    parser.add_argument('--end-epoch', type=int, default=1000, help='training epoch size.')
    parser.add_argument('--lr', type=float, default=0.001, help='start learning rate')
    parser.add_argument('--lr-step', type=int, default=10, help='period of learning rate decay')
    parser.add_argument('--optimizer', default='adam', help='optimizer')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='weight decay')
    parser.add_argument('--mom', type=float, default=0.9, help='momentum')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size in each context')
    parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint')
    parser.add_argument('--pretrained', type=bool, default=False, help='pretrained model')
    parser.add_argument('--data-augumentation', type=bool, default=False, help='is augument data or not')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam.')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for adam.')
    args = parser.parse_args()
    return args


def get_logger():
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s \t%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger


def safe_crop(mat, x, y, crop_size=(im_size, im_size)):
    crop_height, crop_width = crop_size
    if len(mat.shape) == 2:
        ret = np.zeros((crop_height, crop_width), np.uint8)
    else:
        ret = np.zeros((crop_height, crop_width, 3), np.uint8)
    crop = mat[y:y + crop_height, x:x + crop_width]
    h, w = crop.shape[:2]
    ret[0:h, 0:w] = crop
    if crop_size != (im_size, im_size):
        ret = cv.resize(ret, dsize=(im_size, im_size), interpolation=cv.INTER_NEAREST)
    return ret

def gauss(x, sigma):
    y = np.exp(-x ** 2 / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
    return y


def dgauss(x, sigma):
    y = -x * gauss(x, sigma) / (sigma ** 2)
    return y


def gaussgradient(im, sigma):
    epsilon = 1e-2
    halfsize = np.ceil(sigma * np.sqrt(-2 * np.log(np.sqrt(2 * np.pi) * sigma * epsilon))).astype(np.int32)
    size = 2 * halfsize + 1
    hx = np.zeros((size, size))
    for i in range(0, size):
        for j in range(0, size):
            u = [i - halfsize, j - halfsize]
            hx[i, j] = gauss(u[0], sigma) * dgauss(u[1], sigma)

    hx = hx / np.sqrt(np.sum(np.abs(hx) * np.abs(hx)))
    hy = hx.transpose()

    gx = scipy.ndimage.convolve(im, hx, mode='nearest')
    gy = scipy.ndimage.convolve(im, hy, mode='nearest')

    return gx, gy


def compute_gradient_loss(pred, target, trimap):

    pred_x, pred_y = gaussgradient(pred, 1.4)
    target_x, target_y = gaussgradient(target, 1.4)

    pred_amp = np.sqrt(pred_x ** 2 + pred_y ** 2)
    target_amp = np.sqrt(target_x ** 2 + target_y ** 2)

    error_map = (pred_amp - target_amp) ** 2
    loss = np.sum(error_map[trimap == 128])

    return loss / 1000.


def getLargestCC(segmentation):
    labels = label(segmentation, connectivity=1)
    largestCC = labels == np.argmax(np.bincount(labels.flat))
    return largestCC


def compute_connectivity_error(pred, target, trimap, step=0.1):

    # pred = pred / 255.0
    # target = target / 255.0
    h, w = pred.shape

    thresh_steps = list(np.arange(0, 1 + step, step))
    l_map = np.ones_like(pred, dtype=np.float) * -1
    for i in range(1, len(thresh_steps)):
        pred_alpha_thresh = (pred >= thresh_steps[i]).astype(np.int)
        target_alpha_thresh = (target >= thresh_steps[i]).astype(np.int)

        omega = getLargestCC(pred_alpha_thresh * target_alpha_thresh).astype(np.int)
        flag = ((l_map == -1) & (omega == 0)).astype(np.int)
        l_map[flag == 1] = thresh_steps[i - 1]

    l_map[l_map == -1] = 1

    pred_d = pred - l_map
    target_d = target - l_map
    pred_phi = 1 - pred_d * (pred_d >= 0.15).astype(np.int)
    target_phi = 1 - target_d * (target_d >= 0.15).astype(np.int)
    loss = np.sum(np.abs(pred_phi - target_phi)[trimap == 128])

    return loss / 1000.

# alpha prediction loss: the abosolute difference between the ground truth alpha values and the
# predicted alpha values at each pixel. However, due to the non-differentiable property of
# absolute values, we use the following loss function to approximate it.
def mse_core(pred, true, mask):
    return F.mse_loss(pred * mask, true * mask, reduction='sum') / (torch.sum(mask) + epsilon)

def alpha_prediction_loss(y_pred, y_true):
    mask = y_true[:, 1, :, :]
    pred = y_pred[:, :, :]
    true = y_true[:, 0, :, :]
    return mse_core(pred, true, mask)

def composition_loss(y_pred, y_true, image, fg, bg):
    mask = y_true[:, 1:2, :, :]
    mask = torch.cat((mask, mask, mask), dim=1)
    pred = y_pred[:, :, :]
    pred = pred.reshape((-1, 1, pred.shape[1], pred.shape[2]))
    pred = torch.cat((pred, pred, pred), dim=1)
    true = y_true[:, 0, :, :]
    print(pred.shape)
    print(fg.shape)
    print(bg.shape)
    merged = pred * fg + (1 - pred) * bg
    return mse_core(merged, image, mask) / 3.



# compute the MSE error given a prediction, a ground truth and a trimap.
# pred: the predicted alpha matte
# target: the ground truth alpha matte
# trimap: the given trimap
#
def compute_mse(pred, alpha, trimap):
    num_pixels = float((trimap == 128).sum())
    return ((pred - alpha) ** 2).sum() / num_pixels


# compute the SAD error given a prediction and a ground truth.
#
def compute_sad(pred, alpha):
    diff = np.abs(pred - alpha)
    return np.sum(diff) / 1000


def draw_str(dst, target, s):
    x, y = target
    cv.putText(dst, s, (x + 1, y + 1), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=2, lineType=cv.LINE_AA)
    cv.putText(dst, s, (x, y), cv.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv.LINE_AA)


def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

interp_list = [cv.INTER_NEAREST, cv.INTER_LINEAR, cv.INTER_CUBIC, cv.INTER_LANCZOS4]
def maybe_random_interp(cv2_interp):
    if np.random.rand() < 0.5:
        return np.random.choice(interp_list)
    return cv2_interp


num_fgs = 431
num_bgs_per_fg = 100
num_bgs = num_fgs * num_bgs_per_fg
split_ratio = 0.2

out_names_train = 0
out_names_valid = 0

def split_name(split, num, split_index):
    if(split == 'train'):
        names = np.arange(num)
        np.random.shuffle(names)
        names_train = names[:split_index]
        names_valid = names[split_index:]
        global out_names_train 
        out_names_train = names_train
        global out_names_valid
        out_names_valid = names_valid
    return out_names_train, out_names_valid


class InvariantSampler(Sampler):
    def __init__(self, data_source, split, batch_size):
        super().__init__(data_source)
        self.data_source = data_source
        self.split = split
        self.batch_size = batch_size
    def generate(self):
        names_train, names_valid = split_name(self.split, num_fgs * 8, math.ceil(num_fgs * (1-split_ratio)) * 8)
        if self.split == 'train':
            names = names_train
        else:
            names = names_valid
        np.random.shuffle(names)
        names = names * self.batch_size
        names = np.expand_dims(names, 1)
        reduces  = np.copy(names)
        for i in np.arange(1, self.batch_size):
            temp = names + i
            reduces = np.concatenate([reduces, temp], axis=1)
        reduces = reduces.reshape(-1)
        return np.asarray(reduces)
    def __iter__(self):
        self.names = self.generate()
        return iter(self.names)
    def __len__(self):
        return len(self.names)  

class RandomSampler(Sampler):
    def __init__(self, data_source, replacement=False, num_samples=None):
        super().__init__(data_source)
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples

    def num_samples(self):
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self):
        n = self.__len__()
        if self.replacement:
            return iter(torch.randint(high=n, size=(self.num_samples,), dtype=torch.int64).tolist())
        return iter(torch.randperm(n).tolist())

    def __len__(self):
        return self.num_samples()