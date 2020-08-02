import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Conv2d, ReLU, BatchNorm2d, Sequential, Sigmoid
from torchsummaryX import summary

from model import DIMModel, Model, GNModel

# model = Model()
model = GNModel()
# model = DIMModel()
x = torch.Tensor(1, 4, 480, 480)
summary(model, x)
