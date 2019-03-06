## Distance Weighted Sampling

This repo is a pytorch implementation of the ICCV paper  *Sampling Matters in Deep Embedding Learning* . The code is mainly based on [mxnet version](https://github.com/chaoyuaw/incubator-mxnet/tree/master/example/gluon/embedding_learning).



## Usage

See train.sh and train.py



Optional arguments of train.py: 

```
optional arguments:
  -h, --help            show this help message and exit
  --start-epoch N       manual epoch number (useful on restarts)
  --workers N           number of data loading workers (default: 4)
  --data-path DATA_PATH
                        path of data, which contains train,val subdirectory
  --embed-dim EMBED_DIM
                        dimensionality of image embedding. default is 128.
  --feat-dim FEAT_DIM   dimensionality of base_net output. default is 512.
  --classes CLASSES     number of classes in dataset
  --batch-num BATCH_NUM
                        number of batches in one epoch
  --batch-size BATCH_SIZE
                        total batch_size on all gpus.
  --batch-k BATCH_K     number of images per class in a batch. default is 5.
  --gpus GPUS           list of gpus to use, e.g. 0 or 0,2,5.
  --epochs EPOCHS       number of training epochs. default is 20.
  --lr LR               learning rate. default is 0.0001.
  --lr-beta LR_BETA     learning rate for the beta in margin based loss.
                        default is 0.1.
  --margin MARGIN       margin for the margin based loss. default is 0.2.
  --momentum MOMENTUM   momentum
  --beta BETA           initial value for beta. default is 1.2.
  --nu NU               regularization parameter for beta. default is 0.0.
  --factor FACTOR       learning rate schedule factor. default is 0.5.
  --steps STEPS         epochs to update learning rate. default is 20,40,60.
  --resume RESUME       path to checkpoint
  --wd WD               weight decay rate. default is 0.0001.
  --seed SEED           random seed to use
  --model {alexnet,densenet121,densenet161,densenet169,densenet201,inception_v3,resnet101,resnet152,resnet18,resnet34,resnet50,squeezenet1_0,squeezenet1_1,vgg11,vgg11_bn,vgg13,vgg13_bn,vgg16,vgg16_bn,vgg19,vgg19_bn}
                        type of model to use. see vision_model for options.
  --use-pretrained      enable using pretrained model from gluon.
  --print-freq PRINT_FREQ
                        number of batches to wait before logging.
```



