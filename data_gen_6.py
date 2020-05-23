import os
import math
import numbers
import random
import logging
import numpy as np

import tensorflow as tf
import cv2 as cv
import tfrecord_creator

import torch
from   torch.utils.data import Dataset
from   torch.nn import functional as F
from   torchvision import transforms
from   torch.utils.data import BatchSampler, SequentialSampler

from   utils import safe_crop, parse_args, maybe_random_interp, InvariantSampler
from config import unknown_code, fg_path, bg_path, a_path, num_valid, valid_ratio

global args
args = parse_args()

interp_list = [cv.INTER_NEAREST, cv.INTER_LINEAR, cv.INTER_CUBIC, cv.INTER_LANCZOS4]

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

def maybe_random_interp(cv_interp):
    if args.random_interp:
        return np.random.choice(interp_list)
    else:
        return cv_interp

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

class RandomAffine(object):
    """
    Random affine translation
    """
    def __init__(self, degrees, translate=None, scale=None, shear=None, flip=None, resample=False, fillcolor=0):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            assert isinstance(degrees, (tuple, list)) and len(degrees) == 2, \
                "degrees should be a list or tuple and it must be of length 2."
            self.degrees = degrees

        if translate is not None:
            assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
                "translate should be a list or tuple and it must be of length 2."
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError("translation values should be between 0 and 1")
        self.translate = translate

        if scale is not None:
            assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
                "scale should be a list or tuple and it must be of length 2."
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale

        if shear is not None:
            if isinstance(shear, numbers.Number):
                if shear < 0:
                    raise ValueError("If shear is a single number, it must be positive.")
                self.shear = (-shear, shear)
            else:
                assert isinstance(shear, (tuple, list)) and len(shear) == 2, \
                    "shear should be a list or tuple and it must be of length 2."
                self.shear = shear
        else:
            self.shear = shear

        self.resample = resample
        self.fillcolor = fillcolor
        self.flip = flip

    @staticmethod
    def get_params(degrees, translate, scale_ranges, shears, flip, img_size):
        """Get parameters for affine transformation
        Returns:
            sequence: params to be passed to the affine transformation
        """
        angle = random.uniform(degrees[0], degrees[1])
        if translate is not None:
            max_dx = translate[0] * img_size[0]
            max_dy = translate[1] * img_size[1]
            translations = (np.round(random.uniform(-max_dx, max_dx)),
                            np.round(random.uniform(-max_dy, max_dy)))
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            scale = (random.uniform(scale_ranges[0], scale_ranges[1]),
                     random.uniform(scale_ranges[0], scale_ranges[1]))
        else:
            scale = (1.0, 1.0)

        if shears is not None:
            shear = random.uniform(shears[0], shears[1])
        else:
            shear = 0.0

        if flip is not None:
            flip = (np.random.rand(2) < flip).astype(np.int) * 2 - 1

        return angle, translations, scale, shear, flip

    def __call__(self, sample):
        fg = sample['fg']
        alpha = sample['alpha']
        # fg, alpha = sample['fg'], sample['alpha']
        rows, cols, ch = fg.shape
        if np.maximum(rows, cols) < 1024:
            params = self.get_params((0, 0), self.translate, self.scale, self.shear, self.flip, fg.size)
        else:
            params = self.get_params(self.degrees, self.translate, self.scale, self.shear, self.flip, fg.size)

        center = (cols * 0.5 + 0.5, rows * 0.5 + 0.5)
        M = self._get_inverse_affine_matrix(center, *params)
        M = np.array(M).reshape((2, 3))

        fg = cv.warpAffine(fg, M, (cols, rows),
                            flags=maybe_random_interp(cv.INTER_NEAREST) + cv.WARP_INVERSE_MAP)
        alpha = cv.warpAffine(alpha, M, (cols, rows),
                               flags=maybe_random_interp(cv.INTER_NEAREST) + cv.WARP_INVERSE_MAP)

        sample['fg'], sample['alpha'] = fg, alpha

        return sample


    @ staticmethod
    def _get_inverse_affine_matrix(center, angle, translate, scale, shear, flip):
        # Helper method to compute inverse matrix for affine transformation

        # As it is explained in PIL.Image.rotate
        # We need compute INVERSE of affine transformation matrix: M = T * C * RSS * C^-1
        # where T is translation matrix: [1, 0, tx | 0, 1, ty | 0, 0, 1]
        # C is translation matrix to keep center: [1, 0, cx | 0, 1, cy | 0, 0, 1]
        # RSS is rotation with scale and shear matrix
        # It is different from the original function in torchvision
        # The order are changed to flip -> scale -> rotation -> shear
        # x and y have different scale factors
        # RSS(shear, a, scale, f) = [ cos(a + shear)*scale_x*f -sin(a + shear)*scale_y     0]
        # [ sin(a)*scale_x*f          cos(a)*scale_y             0]
        # [     0                       0                      1]
        # Thus, the inverse is M^-1 = C * RSS^-1 * C^-1 * T^-1

        angle = math.radians(angle)
        shear = math.radians(shear)
        scale_x = 1.0 / scale[0] * flip[0]
        scale_y = 1.0 / scale[1] * flip[1]

        # Inverted rotation matrix with scale and shear
        d = math.cos(angle + shear) * math.cos(angle) + math.sin(angle + shear) * math.sin(angle)
        matrix = [
            math.cos(angle) * scale_x, math.sin(angle + shear) * scale_x, 0,
            -math.sin(angle) * scale_y, math.cos(angle + shear) * scale_y, 0
        ]
        matrix = [m / d for m in matrix]

        # Apply inverse of translation and of center translation: RSS^-1 * C^-1 * T^-1
        matrix[2] += matrix[0] * (-center[0] - translate[0]) + matrix[1] * (-center[1] - translate[1])
        matrix[5] += matrix[3] * (-center[0] - translate[0]) + matrix[4] * (-center[1] - translate[1])

        # Apply center translation: C * RSS^-1 * C^-1 * T^-1
        matrix[2] += center[0]
        matrix[5] += center[1]

        return matrix

class RandomJitter(object):
    """
    Random change the hue of the image
    """

    def __call__(self, sample):
        fg, alpha = sample['fg'], sample['alpha']
        # if alpha is all 0 skip
        if np.all(alpha==0):
            return sample
        # convert to HSV space, convert to float32 image to keep precision during space conversion.
        fg = cv.cvtColor(fg.astype(np.float32)/255.0, cv.COLOR_BGR2HSV)
        # Hue noise
        hue_jitter = np.random.randint(-40, 40)
        fg[:, :, 0] = np.remainder(fg[:, :, 0].astype(np.float32) + hue_jitter, 360)
        # Saturation noise
        sat_bar = fg[:, :, 1][alpha > 0].mean()
        sat_jitter = np.random.rand()*(1.1 - sat_bar)/5 - (1.1 - sat_bar) / 10
        sat = fg[:, :, 1]
        sat = np.abs(sat + sat_jitter)
        sat[sat>1] = 2 - sat[sat>1]
        fg[:, :, 1] = sat
        # Value noise
        val_bar = fg[:, :, 2][alpha > 0].mean()
        val_jitter = np.random.rand()*(1.1 - val_bar)/5-(1.1 - val_bar) / 10
        val = fg[:, :, 2]
        val = np.abs(val + val_jitter)
        val[val>1] = 2 - val[val>1]
        fg[:, :, 2] = val
        # convert back to BGR space
        fg = cv.cvtColor(fg, cv.COLOR_HSV2BGR)
        sample['fg'] = fg*255

        return sample


class RandomHorizontalFlip(object):
    """
    Random flip image and label horizontally
    """
    def __init__(self, prob=0.5):
        self.prob = prob
    def __call__(self, sample):
        fg, alpha = sample['fg'], sample['alpha']
        if np.random.uniform(0, 1) < self.prob:
            fg = cv.flip(fg, 1)
            alpha = cv.flip(alpha, 1)
        sample['fg'], sample['alpha'] = fg, alpha

        return sample


class RandomCrop(object):
    """
    Crop randomly the image in a sample, retain the center 1/4 images, and resize to 'output_size'
    :param output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self):
        self.logger = logging.getLogger("Logger")

    def __call__(self, sample):
        crop_size = sample['size']
        self.output_size = crop_size 
        self.margin = self.output_size[0] // 2
        fg, alpha, trimap = sample['fg'],  sample['alpha'], sample['trimap']
        bg = sample['bg']
        h, w = trimap.shape
        bg = cv.resize(bg, (w, h), interpolation=maybe_random_interp(cv.INTER_CUBIC))
        if w < self.output_size[0]+1 or h < self.output_size[1]+1:
            ratio = 1.1*self.output_size[0]/h if h < w else 1.1*self.output_size[1]/w
            # self.logger.warning("Size of {} is {}.".format(name, (h, w)))
            while h < self.output_size[0]+1 or w < self.output_size[1]+1:
                fg = cv.resize(fg, (int(w*ratio), int(h*ratio)), interpolation=maybe_random_interp(cv.INTER_NEAREST))
                alpha = cv.resize(alpha, (int(w*ratio), int(h*ratio)),
                                   interpolation=maybe_random_interp(cv.INTER_NEAREST))
                trimap = cv.resize(trimap, (int(w*ratio), int(h*ratio)), interpolation=cv.INTER_NEAREST)
                bg = cv.resize(bg, (int(w*ratio), int(h*ratio)), interpolation=maybe_random_interp(cv.INTER_CUBIC))
                h, w = trimap.shape
        small_trimap = cv.resize(trimap, (w//4, h//4), interpolation=cv.INTER_NEAREST)
        unknown_list = list(zip(*np.where(small_trimap[self.margin//4:(h-self.margin)//4,
                                                       self.margin//4:(w-self.margin)//4] == 128)))
        unknown_num = len(unknown_list)
        if len(unknown_list) < 10:
            # self.logger.warning("{} does not have enough unknown area for crop.".format(name))
            left_top = (np.random.randint(0, h-self.output_size[0]+1), np.random.randint(0, w-self.output_size[1]+1))
        else:
            idx = np.random.randint(unknown_num)
            left_top = (unknown_list[idx][0]*4, unknown_list[idx][1]*4)

        fg_crop = fg[left_top[0]:left_top[0]+self.output_size[0], left_top[1]:left_top[1]+self.output_size[1],:]
        alpha_crop = alpha[left_top[0]:left_top[0]+self.output_size[0], left_top[1]:left_top[1]+self.output_size[1]]
        bg_crop = bg[left_top[0]:left_top[0]+self.output_size[0], left_top[1]:left_top[1]+self.output_size[1],:]
        trimap_crop = trimap[left_top[0]:left_top[0]+self.output_size[0], left_top[1]:left_top[1]+self.output_size[1]]

        if len(np.where(trimap==128)[0]) == 0:
            self.logger.error("Does not have enough unknown area for crop. Resized to target size."
                                "left_top: {}".format(left_top))
            fg_crop = cv.resize(fg, self.output_size[::-1], interpolation=maybe_random_interp(cv.INTER_NEAREST))
            alpha_crop = cv.resize(alpha, self.output_size[::-1], interpolation=maybe_random_interp(cv.INTER_NEAREST))
            trimap_crop = cv.resize(trimap, self.output_size[::-1], interpolation=cv.INTER_NEAREST)
            bg_crop = cv.resize(bg, self.output_size[::-1], interpolation=maybe_random_interp(cv.INTER_CUBIC))
            # cv.imwrite('../tmp/tmp.jpg', fg.astype(np.uint8))
            # cv.imwrite('../tmp/tmp.png', (alpha*255).astype(np.uint8))
            # cv.imwrite('../tmp/tmp2.png', trimap.astype(np.uint8))
            # raise ValueError("{} does    not have enough unknown area for crop.".format(name))

        sample['fg'], sample['alpha'], sample['trimap'] = fg_crop, alpha_crop, trimap_crop
        sample['bg'] = bg_crop

        return sample


class Rescale(object):
    """
    Rescale the image in a sample to a given size.
    :param output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, alpha, trimap = sample['image'], sample['alpha'], sample['trimap']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        image = cv.resize(image, (new_w, new_h), interpolation=cv.INTER_LINEAR)
        trimap = cv.resize(trimap, (new_w, new_h), interpolation=cv.INTER_NEAREST)
        alpha = cv.resize(alpha, (new_w, new_h), interpolation=cv.INTER_LINEAR)

        sample['image'], sample['alpha'], sample['trimap'] = image, alpha, trimap

        return sample


class OriginScale(object):
    def __call__(self, sample):
        h, w = sample["alpha_shape"]
        # sample['origin_trimap'] = sample['trimap']
        # # if h % 32 == 0 and w % 32 == 0:
        # #     return sample
        # # target_h = h - h % 32
        # # target_w = w - w % 32
        # target_h = 32 * ((h - 1) // 32 + 1)
        # target_w = 32 * ((w - 1) // 32 + 1)
        # sample['image'] = cv.resize(sample['image'], (target_w, target_h), interpolation=cv.INTER_CUBIC)
        # sample['trimap'] = cv.resize(sample['trimap'], (target_w, target_h), interpolation=cv.INTER_NEAREST)

        if h % 32 == 0 and w % 32 == 0:
            return sample
        target_h = 32 * ((h - 1) // 32 + 1)
        target_w = 32 * ((w - 1) // 32 + 1)
        pad_h = target_h - h
        pad_w = target_w - w

        padded_image = np.pad(sample['image'], ((0,pad_h), (0, pad_w), (0,0)), mode="reflect")
        padded_trimap = np.pad(sample['trimap'], ((0,pad_h), (0, pad_w)), mode="reflect")

        sample['image'] = padded_image
        sample['trimap'] = padded_trimap

        return sample


class GenTrimap(object):
    def __init__(self):
        self.erosion_kernels = [None] + [cv.getStructuringElement(cv.MORPH_ELLIPSE, (size, size)) for size in range(1,30)]

    def __call__(self, sample):
        alpha = sample['alpha']
        # Adobe 1K
        fg_width = np.random.randint(1, 30)
        bg_width = np.random.randint(1, 30)
        fg_mask = (alpha + 1e-5).astype(np.int).astype(np.uint8)
        bg_mask = (1 - alpha + 1e-5).astype(np.int).astype(np.uint8)
        fg_mask = cv.erode(fg_mask, self.erosion_kernels[fg_width])
        bg_mask = cv.erode(bg_mask, self.erosion_kernels[bg_width])

        trimap = np.ones_like(alpha) * 128
        trimap[fg_mask == 1] = 255
        trimap[bg_mask == 1] = 0

        sample['trimap'] = trimap
        return sample


class Composite(object):
    def __call__(self, sample):
        fg, bg, alpha = sample['fg'], sample['bg'], sample['alpha']
        alpha[alpha < 0 ] = 0
        alpha[alpha > 1] = 1
        fg[fg < 0 ] = 0
        fg[fg > 255] = 255
        bg[bg < 0 ] = 0
        bg[bg > 255] = 255

        image = fg * alpha[:, :, None] + bg * (1 - alpha[:, :, None])
        sample['image'] = image
        return sample

class ToTensor(object):
    """
    Convert ndarrays in sample to Tensors with normalization.
    """
    def __init__(self, split='train'):
        self.split = split

    def __call__(self, sample):
        image = sample['image']
        image = transforms.ToTensor()(image)
        image = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)
        sample['image'] = image
        return sample

class DIMDataset(Dataset):
    def __init__(self, split="train", test_scale="resize"):
        self.split = split

        if args.data_augumentation:
            train_trans = [
                            RandomAffine(degrees=30, scale=[0.8, 1.25], shear=10, flip=0.5),
                            GenTrimap(),
                            RandomCrop(),
                            RandomJitter(),
                            Composite(),
                            ToTensor(split="train"),]
        else:
            train_trans = [ GenTrimap(),
                            RandomCrop(),
                            Composite(),
                            ToTensor(split="train") ]

        if test_scale.lower() == "origin":
            test_trans = [ OriginScale(), ToTensor() ]
        elif test_scale.lower() == "resize":
            test_trans = [ Rescale((320,320)), ToTensor() ]
        elif test_scale.lower() == "crop":
            test_trans = [ RandomCrop(), ToTensor() ]
        else:
            raise NotImplementedError("test_scale {} not implemented".format(test_scale))

        self.transform = {
            'train':
                transforms.Compose(train_trans),
            'valid':
                transforms.Compose([
                    OriginScale(),
                    ToTensor(split='valid'),
                ]),
            'test':
                transforms.Compose(test_trans)
        }[split]

        self.erosion_kernels = [None] + [cv.getStructuringElement(cv.MORPH_ELLIPSE, (size, size)) for size in range(1,20)]
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

        self.current_index = -1
        self.current_fg = None
        self.current_alpha = None
        self.is_resize = False

    def __getitem__(self, i):
        fcount = self.fgs[i]
        print(i)
        if i % args.batch_size == 0:
            self.current_index = fcount
            alpha = get_raw("a", fcount)
            alpha = np.reshape(alpha, (alpha.shape[0], alpha.shape[1])).astype(np.float32) / 255.
            fg = get_raw("fg", fcount)
            if args.data_augumentation:
                fg, alpha = self._composite_fg(fg, alpha, i)
            self.current_fg = fg
            self.current_alpha = alpha
            different_sizes = [(320, 320), (480, 480), (640, 640), (512, 512)]
            crop_size = random.choice(different_sizes)
            self.crop_size = crop_size 
            # self.is_resize = True if np.random.rand() < 0.25 else False
        else:
            fg = self.current_fg
            alpha = self.current_alpha
            crop_size = self.crop_size
        
        bcount = np.random.randint(num_bgs)
        bg = get_raw("bg", bcount)
        sample = {'fg': fg, 'alpha': alpha, 'bg': bg, 'size': crop_size, 'alpha_shape': alpha.shape}
        # print(i)
        sample = self.transform(sample)
        img, trimap, alpha = sample['image'], sample['trimap'], sample['alpha']
        im_size = img.shape[1]
        # crop size 320:640:480 = 1:1:1

        x = torch.zeros((4, im_size, im_size), dtype=torch.float)
        x[0:3, :, :] = img
        x[3, :, :] = torch.from_numpy(trimap.copy())

        y = np.empty((2, im_size, im_size), dtype=np.float32)
        y[0, :, :] = alpha
        mask = np.equal(trimap, 128).astype(np.float32)
        y[1, :, :] = mask

        return x, y

    def _composite_fg(self, fg, alpha, idx):

        if np.random.rand() < 0.5:
            idx2 = np.random.randint(self.fg_num_unique) + idx
            idx2 = idx2 % self.fg_num_unique
            fg2 = get_raw("fg", idx2)
            alpha2 = get_raw("a", idx2)
            alpha2 = np.reshape(alpha2, (alpha2.shape[0], alpha2.shape[1]))
            alpha2 = alpha2.astype(np.float32) / 255.0
            h, w = alpha.shape
            fg2 = cv.resize(fg2, (w, h), interpolation=maybe_random_interp(cv.INTER_NEAREST))
            alpha2 = cv.resize(alpha2, (w, h), interpolation=maybe_random_interp(cv.INTER_NEAREST))

            alpha_tmp = 1 - (1 - alpha) * (1 - alpha2)
            if  np.any(alpha_tmp < 1):
                fg = fg.astype(np.float32) * alpha[:,:,None] + fg2.astype(np.float32) * (1 - alpha[:,:,None])
                # The overlap of two 50% transparency should be 25%
                alpha = alpha_tmp
                fg = fg.astype(np.uint8)

        if np.random.rand() < 0.25:
            fg = cv.resize(fg, (640, 640), interpolation=maybe_random_interp(cv.INTER_NEAREST))
            alpha = cv.resize(alpha, (640, 640), interpolation=maybe_random_interp(cv.INTER_NEAREST))

        return fg, alpha

    def __len__(self):
        return len(self.fgs)

num_fgs = 431
num_bgs_per_fg = 100
num_bgs = num_fgs * num_bgs_per_fg
split_ratio = 0.2

def split_name():
    names = list(range(num_fgs))
    np.random.shuffle(names)
    split_index = math.ceil(num_fgs - num_fgs * valid_ratio)
    names_train = np.copy(names)
    names_valid = np.copy(names)
    return names_train, names_valid

# if __name__ == '__main__':

#     from torch.utils.data import DataLoader

#     logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(levelname)s: %(message)s', datefmt='%m-%d %H:%M:%S')

#     args.data_augmentation = True
#     data_dataset = DIMDataset(split='train')
#     batch_size = 2
#     num_workers = 0
#     train_batch_sample = BatchSampler(InvariantSampler(data_dataset, "train", args.batch_size), batch_size=args.batch_size,drop_last=False)
#     data_loader = DataLoader(
#         data_dataset, batch_sampler=train_batch_sample, num_workers=num_workers)
#     import time

#     mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
#     std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
#     t = time.time()
#     from tqdm import tqdm

#     # for i, batch in enumerate(tqdm(data_loader)):
#     #     image, bg, alpha = batch['image'], batch['bg'], batch['alpha']
#     b = next(iter(data_loader))
#     for i in range(b['image'].shape[0]):
#         image = (b['image'][i]*std+mean).data.numpy()*255
#         image = image.transpose(1,2,0)[:,:,::-1]
#         trimap = b['trimap'][i].argmax(dim=0).data.numpy()*127
#         cv.imwrite('./tmp/'+str(i)+'.jpg', image.astype(np.uint8))
#         cv.imwrite('./tmp/'+str(i)+'.png', trimap.astype(np.uint8))
#         # if i > 10:
#         #     break
#         # print(b['image_name'][i])
#     # print(time.time() - t, 'seconds', 'batch_size', batch_size, 'num_workers', num_workers)