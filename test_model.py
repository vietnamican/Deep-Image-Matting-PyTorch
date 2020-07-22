import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Conv2d, ReLU, BatchNorm2d, Sequential, Sigmoid
from torchsummaryX import summary

from model import DIMModel

model = DIMModel()
x = torch.Tensor(1, 4, 1000, 999)
summary(model, x)
