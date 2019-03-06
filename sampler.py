import torch
import torchvision
import random
from torch.utils.data.sampler import  Sampler
from scipy.special import comb
import numpy as np

class BalancedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, batch_k, length=None):
        assert (batch_size % batch_k == 0 ) and (batch_size > 0)
        self.dataset = {}
        self.balanced_max = 0
        self.batch_size = batch_size
        self.batch_k = batch_k
        self.length = length

        # Save all the indices for all the classes
        for idx in range(0, len(dataset)):
            label = self._get_label(dataset, idx)
            if label not in self.dataset:
                self.dataset[label] = []
            self.dataset[label].append(idx)
        
        num_samples = [len(value) for value in self.dataset.values()]
        self.max_samples = max(num_samples)
        self.min_samples = min(num_samples)

        assert self.min_samples >= self.batch_k
    
        self.keys = list(self.dataset.keys())
        #self.currentkey = 0

    def __iter__(self):
        while(True):
            batch = []
            classes = np.random.choice(range(len(self.keys)), size=int(self.batch_size/self.batch_k), replace=False )
            for cls in classes:
                cls_idxs = self.dataset[self.keys[cls]]
                for k in np.random.choice(range(len(cls_idxs)), size=self.batch_k, replace=False):
                    batch.append(cls_idxs[k])
            yield batch
    def __len__(self):
        if self.length is not None:
            return self.length
        return int(len(self.keys) * (comb(self.max_samples, self.batch_k) + comb(self.min_samples, self.batch_k))/2)
        
    def _get_label(self, dataset, idx):
        dataset_type = type(dataset)
        if dataset_type is torchvision.datasets.MNIST:
            return dataset.train_labels[idx].item()
        elif dataset_type is torchvision.datasets.ImageFolder:
            return dataset.imgs[idx][1]
        else:
            raise NotImplementedError
        
	
