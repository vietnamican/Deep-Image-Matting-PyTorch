import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn
import random

from config import im_size, grad_clip
from data_gen_unet_7_b1 import DIMDataset
from model import Model
from utils import parse_args, save_checkpoint, AverageMeter, clip_gradient, get_logger, get_learning_rate, \
    alpha_prediction_loss, adjust_learning_rate


def train_net(args):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    checkpoint = args.checkpoint
    start_epoch = 1
    best_loss = float('inf')
    writer = SummaryWriter(logdir=args.logdir)
    epochs_since_improvement = 0
    decays_since_improvement = 0

    # Initialize / load checkpoint
    if checkpoint is None:
        torch.random.manual_seed(7)
        torch.cuda.manual_seed(7)
        np.random.seed(7)
        random.seed(7)
        model = Model()
        model.to(args.device)
        if args.device == 'cuda':
            model = nn.DataParallel(model)

        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.mom,
                                        weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        model = Model()
        model.to(args.device)
        if args.device == 'cuda':
            model.module.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint['model'])
        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.mom,
                                        weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer.load_state_dict(checkpoint['optimizer'])
        if 'torch_seed' in checkpoint:
            torch.random.set_rng_state(checkpoint['torch_seed'])
        else:
            torch.random.manual_seed(7)
        if 'torch_cuda_seed' in checkpoint:
            torch.cuda.set_rng_state(checkpoint['torch_cuda_seed'])
        else:
            torch.cuda.manual_seed(7)
        if 'np_seed' in checkpoint:
            np.random.set_state(checkpoint['np_seed'])
        else:
            np.random.seed(7)
        if 'python_seed' in checkpoint:
            random.setstate(checkpoint['python_seed'])
        else:
            random.seed(7)

    logger = get_logger()

    # Custom dataloaders
    train_dataset = DIMDataset('train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    valid_dataset = DIMDataset('valid')
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.val_batch_size, shuffle=False,
                                               num_workers=8)

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
                           logger=logger)
        effective_lr = get_learning_rate(optimizer)
        print('Current effective learning rate: {}\n'.format(effective_lr))

        writer.add_scalar('Train_Loss', train_loss, epoch)
        writer.add_scalar('Learning_Rate', effective_lr, epoch)

        # One epoch's validation
        valid_loss = valid(valid_loader=valid_loader,
                           model=model,
                           epoch=epoch,
                           logger=logger)

        writer.add_scalar('Valid_Loss', valid_loss, epoch)

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
        save_checkpoint(epoch, epochs_since_improvement, model, optimizer, best_loss, is_best, args.checkpointdir)


def train(train_loader, model, optimizer, epoch, logger):
    model.train()  # train mode (dropout and batchnorm is used)

    losses = AverageMeter()

    # Batches
    for i, (img, alpha_label) in enumerate(train_loader):
        # Move to GPU, if available
        img = img.type(torch.FloatTensor).to(args.device)  # [N, 4, 320, 320]
        alpha_label = alpha_label.type(torch.FloatTensor).to(args.device)  # [N, 320, 320]
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

        if i % args.print_freq == 0:
            status = 'Epoch: [{0}][{1}/{2}]\t' \
                     'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader), loss=losses)
            logger.info(status)

    return losses.avg


def valid(valid_loader, model, epoch, logger):
    model.eval()  # eval mode (dropout and batchnorm is NOT used)

    losses = AverageMeter()

    # Batches
    for i, (img, alpha_label) in enumerate(valid_loader):
        # Move to GPU, if available
        img = img.type(torch.FloatTensor).to(args.device)  # [N, 3, 320, 320]
        alpha_label = alpha_label.type(torch.FloatTensor).to(args.device)  # [N, 320, 320]
        alpha_label = alpha_label.reshape((-1, 2, im_size * im_size))  # [N, 320*320]

        # Forward prop.
        alpha_out = model(img)  # [N, 320, 320]
        alpha_out = alpha_out.reshape((-1, 1, im_size * im_size))  # [N, 320*320]

        # Calculate loss
        # loss = criterion(alpha_out, alpha_label)
        loss = alpha_prediction_loss(alpha_out, alpha_label)

        # Keep track of metrics
        losses.update(loss.item())

        if i % args.print_freq == 0:
            status = 'Epoch: [{0}][{1}/{2}]\t' \
                     'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(valid_loader), loss=losses)
            logger.info(status)

    # Print status
    status = 'Validation: Loss {loss.avg:.4f}\n'.format(loss=losses)

    logger.info(status)

    return losses.avg


def main():
    global args
    args = parse_args()
    train_net(args)


if __name__ == '__main__':
    main()
