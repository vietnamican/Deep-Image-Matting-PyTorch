import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # sets device for model and PyTorch tensors

im_size = 320
unknown_code = 128
epsilon = 1e-6
epsilon_sqr = epsilon ** 2

num_samples = 43100
num_train = 34480
# num_samples - num_train_samples
num_valid = 8620

num_fgs = 481
num_bgs_per_fg = 100
num_bgs = 44100

valid_ratio = 0.2

# Training parameters
num_workers = 1  # for data-loading; right now, only 1 works with h5py
grad_clip = 5.  # clip gradients at an absolute value of
checkpoint = None  # path to checkpoint, None if none

##############################################################
# Set your paths here

# path to provided foreground images
fg_path = 'data/fg/'

# path to provided alpha mattes
a_path = 'data/mask/'

# Path to background images (MSCOCO)
bg_path = 'data/bg/'

# Path to folder where you want the composited images to go
out_path = 'data/merged/'

max_size = 1600
fg_path_test = 'data/fg_test/'
a_path_test = 'data/mask_test/'
bg_path_test = 'data/bg_test/'
out_path_test = 'data/merged_test/'
fg_path_test_composition = 'data/fg_test_composition/'
a_path_test_composition = 'data/mask_test_composition/'
bg_path_test_composition = 'data/bg_test_composition/'
out_path_test_composition = 'data/merged_test_composition/'
trimap_path_test_composition = 'data/trimap_test_composition/'
##############################################################
