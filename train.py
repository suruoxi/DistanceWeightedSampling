import argparse
import logging
import time
import os

import numpy as np
from bottleneck import argpartition

import models
import torch
from torch import nn

logging.basicConfig(level=logging.INFO)

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


parser = argparse.ArgumentParser(description='train a model for image classification.')
parser.add_argument('--data-path', type=str, default='data/CUB_200_2011',
                    help='path of data.')
parser.add_argument('--embed-dim', type=int, default=128,
                    help='dimensionality of image embedding. default is 128.')
parser.add_argument('--feat-dim', type=int, default=512,
                    help='dimensionality of base_net output. default is 512.')
parser.add_argument('--batch-size', type=int, default=70,
                    help='training batch size per device (CPU/GPU). default is 70.')
parser.add_argument('--batch-k', type=int, default=5,
                    help='number of images per class in a batch. default is 5.')
parser.add_argument('--gpus', type=str, default='',
                    help='list of gpus to use, e.g. 0 or 0,2,5. empty means using cpu.')
parser.add_argument('--epochs', type=int, default=20,
                    help='number of training epochs. default is 20.')
parser.add_argument('--optimizer', type=str, default='adam',
                    help='optimizer. default is adam.')
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
parser.add_argument('--steps', type=str, default='12,14,16,18',
                    help='epochs to update learning rate. default is 12,14,16,18.')
parser.add_argument('--wd', type=float, default=0.0001,
                    help='weight decay rate. default is 0.0001.')
parser.add_argument('--seed', type=int, default=123,
                    help='random seed to use. default=123.')
parser.add_argument('--model', type=str, default='resnet50',choices=model_names,
                    help='type of model to use. see vision_model for options.')
parser.add_argument('--save-model-prefix', type=str, default='margin_loss_model',
                    help='prefix of models to be saved.')
parser.add_argument('--use-pretrained', action='store_true',
                    help='enable using pretrained model from gluon.')
parser.add_argument('--kvstore', type=str, default='device',
                    help='kvstore to use for trainer.')
parser.add_argument('--log-interval', type=int, default=20,
                    help='number of batches to wait before logging.')
args = parser.parse_args()

logging.info(opt)

# gpus setting
os.environ['VISIBLE_CUDA_DEVICES'] = args.gpus

# construct model

if not opt.use_pretained:
    model = models.__dict__[args.model](num_classes=args.feat_dim)
else:
    model = models.__dict__[args.model](pretrained=True)
    try:
        model.fc = nn.Linear(model.fc.in_features, args.feat_dim)
    except NameError as e:
        print("Error: current works only with model having fc layer as the last layer, try modify the code")
        exit(-1)

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

def evaluate_emb(emb, labels):
    """Evaluate embeddings based on Recall@k."""
    d_mat = get_distance_matrix(emb)
    d_mat = d_mat.asnumpy()
    labels = labels.asnumpy()

    names = []
    accs = []
    for k in [1, 2, 4, 8, 16]:
        names.append('Recall@%d' % k)
        correct, cnt = 0.0, 0.0
        for i in range(emb.shape[0]):
            d_mat[i, i] = 1e10
            nns = argpartition(d_mat[i], k)[:k]
            if any(labels[i] == labels[nn] for nn in nns):
                correct += 1
            cnt += 1
        accs.append(correct/cnt)
    return names, accs












