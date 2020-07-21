import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Conv2d, ReLU, BatchNorm2d
from torchsummaryX import summary

from model import DIMModel, UpBlock

if __name__ == "__main__":
    model = DIMModel()
    input = torch.Tensor(1, 4, 320, 320)
    summary(model, input)

    # model = UpBlock(20, 512, 256, 3, 1, 1)
    # input = torch.Tensor(1, 512, 10, 10)
    # summary(model, input)
