import torch
import torchvision
import random
import torch.utils.data.sampler.Sampler as Sampler
from scipy.special import comb

class BalancedBatchSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, batch_size, batch_k, lenght=None):
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
            classes = np.random.choice(range(len(self.keys)), size=self.batch_size/self.batch_k, replace=False )
            for cls in classes:
                cls_idxs = self.dataset[self.keys[cls]]
                for k in np.random.choice(range(len(cls_idxs)), size=self.batch_k, replace=False):
                    batch.append(cls_idxs[k])
            yield batch
    def __len__(self):
        if self.length is not None:
            return self.length
        return int(len(self.keys) * (comb(self.max_samples, self.batch_k) + comb(self.min_samples, self.batch_k))/2)
        
        
	
