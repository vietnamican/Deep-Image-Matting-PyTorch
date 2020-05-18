import torch
import torch.nn as nn

import models
import models_v16
from config import device

# old_model = models.DIMModel()
# new_model = models_v16.DIMModel()



def migrate(new_model):
    # print(new_model)
    checkpoint = 'BEST_checkpoint_older.tar'
    checkpoint = torch.load(checkpoint)
    old_model = checkpoint['model'].module
    # print(dict(old_model.named_parameters()))
    # print("old")
    # print(dict(old_model.down1.conv1.cbr_unit[1].named_parameters()))
    # print("new")
    # print(dict(new_model.down1.conv1.cbr_unit[1].named_parameters()))
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
        # if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
        if l1.weight.size() == l2.weight.size() and l1.bias.size() == l2.bias.size():
            print("success")
            l2.weight.data.copy_(l1.weight.data)
            # l2.weight.data = l1.weight.data
            l2.bias.data.copy_(l1.bias.data)
            # l2.bias.data = l1.bias.data
    del checkpoint
    # print("old")
    # print(dict(old_model.down1.conv1.cbr_unit[1].named_parameters()))
    # print("new")
    # print(dict(new_model.down1.conv1.cbr_unit[1].named_parameters()))



if __name__ == "__main__":
    migrate(models_v16.DIMModel())