import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl

from config import im_size, grad_clip, print_freq
from data_gen_2 import DIMDataset
from models_v16 import DIMModel
from utils import parse_args, save_checkpoint, AverageMeter, clip_gradient, get_logger, get_learning_rate, \
    alpha_prediction_loss, adjust_learning_rate


def train_net(index, args):
    torch.manual_seed(7)
    np.random.seed(7)
    checkpoint = args.checkpoint
    start_epoch = 0
    best_loss = float('inf')
    # writer = SummaryWriter()
    epochs_since_improvement = 0
    decays_since_improvement = 0

    # Initialize / load checkpoint
    if checkpoint is None:
        model = DIMModel(n_classes=1, in_channels=4, is_unpooling=True, pretrain=True)
        model = nn.DataParallel(model)

        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.mom,
                                        weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    else:
        checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    logger = get_logger()
    device = xm.xla_device()
    # Move to GPU, if available
    model = model.to(device)

    # Custom dataloaders
    train_dataset = DIMDataset('train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_dataset = DIMDataset('valid')
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

    # Epochs
    for epoch in range(start_epoch, args.end_epoch):
        if args.optimizer == 'sgd' and epochs_since_improvement == 10:
            break

        if args.optimizer == 'sgd' and epochs_since_improvement > 0 and epochs_since_improvement % 2 == 0:
            checkpoint = 'BEST_checkpoint.tar'
            checkpoint = torch.load(checkpoint)
            model = checkpoint['model']
            optimizer = checkpoint['optimizer']
            decays_since_improvement += 1
            print("\nDecays since last improvement: %d\n" % (decays_since_improvement,))
            adjust_learning_rate(optimizer, 0.6 ** decays_since_improvement)

        # One epoch's training
        train_loss = train(train_loader=train_loader,
                           model=model,
                           optimizer=optimizer,
                           epoch=epoch,
                           logger=logger,
                           device=device)
        effective_lr = get_learning_rate(optimizer)
        print('Current effective learning rate: {}\n'.format(effective_lr))

        # writer.add_scalar('Train_Loss', train_loss, epoch)
        # writer.add_scalar('Learning_Rate', effective_lr, epoch)

        # One epoch's validation
        valid_loss = valid(valid_loader=valid_loader,
                           model=model,
                           logger=logger,
                           device=device)

        # writer.add_scalar('Valid_Loss', valid_loss, epoch)

        # Check if there was an improvement
        is_best = valid_loss < best_loss
        best_loss = min(valid_loss, best_loss)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0
            decays_since_improvement = 0

        # Save checkpoint
        save_checkpoint(epoch, epochs_since_improvement, model, optimizer, best_loss, is_best)


def train(train_loader, model, optimizer, epoch, logger, device):
    model.train()  # train mode (dropout and batchnorm is used)

    losses = AverageMeter()
    with torch.no_grad():
        para_train_loader = pl.ParallelLoader(train_loader, [device]).per_device_loader(device)
        for i, batch in enumerate(para_train_loader):
            img, alpha_label = batch
    # # Batches
    # for i, (img, alpha_label) in enumerate(train_loader):
            # Move to GPU, if available
            img = img.type(torch.FloatTensor).to(device)  # [N, 4, 320, 320]
            alpha_label = alpha_label.type(torch.FloatTensor).to(device)  # [N, 320, 320]
            alpha_label = alpha_label.reshape((-1, 2, im_size * im_size))  # [N, 320*320]

            # Forward prop.
            alpha_out = model(img)  # [N, 3, 320, 320]
            alpha_out = alpha_out.reshape((-1, 1, im_size * im_size))  # [N, 320*320]

            # Calculate loss
            # loss = criterion(alpha_out, alpha_label)
            loss = alpha_prediction_loss(alpha_out, alpha_label)

            # Back prop.
            optimizer.zero_grad()
            loss.backward()

            # Clip gradients
            clip_gradient(optimizer, grad_clip)

            # Update weights
            optimizer.step()

            # Keep track of metrics
            losses.update(loss.item())

            # Print status

            if i % print_freq == 0:
                status = 'Epoch: [{0}][{1}/{2}]\t' \
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader), loss=losses)
                logger.info(status)

    return losses.avg


def valid(valid_loader, model, logger, device):
    model.eval()  # eval mode (dropout and batchnorm is NOT used)

    losses = AverageMeter()

    with torch.no_grad():
        para_valid_loader = pl.ParallelLoader(valid_loader, [device]).per_device_loader(device)
        for i, batch in enumerate(para_valid_loader):
            img, alpha_label = batch
    # # Batches
    # for img, alpha_label in valid_loader:
            # Move to GPU, if available
            img = img.type(torch.FloatTensor).to(device)  # [N, 3, 320, 320]
            alpha_label = alpha_label.type(torch.FloatTensor).to(device)  # [N, 320, 320]
            alpha_label = alpha_label.reshape((-1, 2, im_size * im_size))  # [N, 320*320]

            # Forward prop.
            alpha_out = model(img)  # [N, 320, 320]
            alpha_out = alpha_out.reshape((-1, 1, im_size * im_size))  # [N, 320*320]

            # Calculate loss
            # loss = criterion(alpha_out, alpha_label)
            loss = alpha_prediction_loss(alpha_out, alpha_label)

            # Keep track of metrics
            losses.update(loss.item())

    # Print status
    status = 'Validation: Loss {loss.avg:.4f}\n'.format(loss=losses)

    logger.info(status)

    return losses.avg


def main():
    global args
    args = parse_args()
    # train_net(args)
    xmp.spawn(train_net, args=(args,), nprocs=8, start_method='fork')


if __name__ == '__main__':
    main()
