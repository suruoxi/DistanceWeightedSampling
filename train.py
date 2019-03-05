import argparse
import logging
import time
import os

import numpy as np
from bottleneck import argpartition

import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch import nn
from torch.autograd import Variable
from torch.nn import Parameter

from model import *
from sampler import BalancedBatchSampler 


logging.basicConfig(level=logging.INFO)

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


parser = argparse.ArgumentParser(description='train a model for image classification.')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--data-path', type=str, default='data/CUB_200_2011',
                    help='path of data.')
parser.add_argument('--embed-dim', type=int, default=128,
                    help='dimensionality of image embedding. default is 128.')
parser.add_argument('--feat-dim', type=int, default=512,
                    help='dimensionality of base_net output. default is 512.')
parser.add_argument('--classes', type=int, required=True,
                    help='number of classes in dataset')
parser.add_argument('--batch-size', type=int, default=70,
                    help='training batch size per device (CPU/GPU). default is 70.')
parser.add_argument('--batch-k', type=int, default=5,
                    help='number of images per class in a batch. default is 5.')
parser.add_argument('--gpus', type=str, default='',
                    help='list of gpus to use, e.g. 0 or 0,2,5. empty means using cpu.')
parser.add_argument('--epochs', type=int, default=80,
                    help='number of training epochs. default is 20.')
#parser.add_argument('--optimizer', type=str, default='adam',
#                    help='optimizer. default is adam.')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate. default is 0.0001.')
parser.add_argument('--lr-beta', type=float, default=0.1,
                    help='learning rate for the beta in margin based loss. default is 0.1.')
parser.add_argument('--margin', type=float, default=0.2,
                    help='margin for the margin based loss. default is 0.2.')
parser.add_argument('--beta', type=float, default=1.2,
                    help='initial value for beta. default is 1.2.')
parser.add_argument('--nu', type=float, default=0.0,
                    help='regularization parameter for beta. default is 0.0.')
parser.add_argument('--factor', type=float, default=0.5,
                    help='learning rate schedule factor. default is 0.5.')
parser.add_argument('--steps', type=str, default='20,40,60',
                    help='epochs to update learning rate. default is 12,14,16,18.')
parser.add_argument('--wd', type=float, default=0.0001,
                    help='weight decay rate. default is 0.0001.')
parser.add_argument('--seed', type=int, default=123,
                    help='random seed to use. default=123.')
parser.add_argument('--model', type=str, default='resnet50',choices=model_names,
                    help='type of model to use. see vision_model for options.')
#parser.add_argument('--save-model-prefix', type=str, default='margin_loss_model',
#                    help='prefix of models to be saved.')
parser.add_argument('--use-pretrained', action='store_true',
                    help='enable using pretrained model from gluon.')
#parser.add_argument('--kvstore', type=str, default='device',
#                    help='kvstore to use for trainer.')
parser.add_argument('--print-freq', type=int, default=20,
                    help='number of batches to wait before logging.')
args = parser.parse_args()

logging.info(args)

# gpus setting
os.environ['VISIBLE_CUDA_DEVICES'] = args.gpus

# construct model

if not args.use_pretained:
    model = models.__dict__[args.model](num_classes=args.feat_dim)
else:
    model = models.__dict__[args.model](pretrained=True)
    try:
        model.fc = nn.Linear(model.fc.in_features, args.feat_dim)
    except NameError as e:
        print("Error: current works only with model having fc layer as the last layer, try modify the code")
        exit(-1)


model = MarginNet(base_net=model, emb_dim=args.emb_dim, batch_k=args.batch_k, in_dim=args.feat_dim)

criterion = MarginLoss(margin=args.margin,nu=args.nu)

optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, 
        weight_decay = args.weight_decay)

beta = Parameter(torch.ones((args.classes,), dtype=torch.float32)*args.beta)

optimizer_beta = torch.optim.SGD(beta, args.lr_beta, momentum=args.momentum, weight_decay=args.weight_decay)

if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_acc1 = checkpoint['best_acc1']
        if args.gpu is not None:
            # best_acc1 may be from a checkpoint from a different GPU
            best_acc1 = best_acc1.to(args.gpu)
        state_dict = {}
        for k,v in checkpoint['state_dict'].items():
            if k.startswith('module.'):
                k = k[7:]
                state_dict[k] = v
        model.load_state_dict(state_dict)
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))


model.cuda()


# dataset 
traindir = os.path.join(args.data_path, 'train')
valdir = os.path.join(args.data_path, 'val')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

train_dataset = datasets.ImageFolder(
    traindir,
    transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])
    )

batch_sampler = BalancedBatchSampler(train_dataset, args.batch_size, args.batch_k, length=100000)

train_loader = torch.utils.data.DataLoader(
    batch_sampler=batch_sampler,
    dataset=train_dataset,
    num_workers=args.workers,
    pin_memory=True
    )


#def evaluate_emb(emb, labels):
#    """Evaluate embeddings based on Recall@k."""
#    d_mat = get_distance_matrix(emb)
#    d_mat = d_mat.asnumpy()
#    labels = labels.asnumpy()
#
#    names = []
#    accs = []
#    for k in [1, 2, 4, 8, 16]:
#        names.append('Recall@%d' % k)
#        correct, cnt = 0.0, 0.0
#        for i in range(emb.shape[0]):
#            d_mat[i, i] = 1e10
#            nns = argpartition(d_mat[i], k)[:k]
#            if any(labels[i] == labels[nn] for nn in nns):
#                correct += 1
#            cnt += 1
#        accs.append(correct/cnt)
#    return names, accs
#

def train(train_loader, model, criterion, optimizer, optimizer_beta, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (x,y) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            x = x.cuda(args.gpu, non_blocking=True)
        y = y.cuda()

        # compute output
        a_indices, anchors, positives, negatives, _ = model(x)
        if args.lr_beta > 0.0:
            loss = criterion(anchors, positives, negatives, beta, y[a_indices])
        else:
            loss = criterion(anchors, positives, negatives, args.beta, None)

        # measure accuracy and record loss
        losses.update(loss.item(), x.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        optimizer_beta.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    steps = [int(step) for step in args.steps.split(',')]
    #lr = args.lr * (0.1 ** (epoch // 30))
    if epoch in steps:
        for param_group in optimizer.param_groups:
            #param_group['lr'] = lr
            param_group['lr'] *= args.factor

class AverageMeter(object):
    """Computes and stores the average and current value"""
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


if __name__ == "__main__":
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)
        adjust_learning_rate(optimizer_beta, epoch, args)
        
        # train for one epoch
        train(train_loader, model, criterion, optimizer, optimizer_beta,  epoch, args)

        # evaluate
        # 

        state = {
            'epoch': epoch+1,
            'arch': args.model,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
            }
        torch.save(state, 'checkpoint_%d.pth.tar'%(epoch+1))









