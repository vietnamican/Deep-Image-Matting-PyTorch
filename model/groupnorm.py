import torch
import torch.nn as nn
from torchsummaryX import summary


class GroupNorm(nn.Module):
    def __init__(self, num_channels, eps=1e-05, affine=True):
        super(GroupNorm, self).__init__()
        num_groups = num_channels // 8 if num_channels < 256 else 32
        self.groupnorm = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels, eps=eps, affine=affine)

    def forward(self, x):
        return self.groupnorm(x)
