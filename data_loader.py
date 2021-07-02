# Following links are what we referred to for data generation
#https://github.com/huaxiuyao/HSML/blob/master/data_generator.py
#https://github.com/timchen0618/pytorch-leo/blob/master/data.py

import numpy as np
import os
import random
from scipy import signal
from PIL import Image
import pdb
import itertools
import pickle
import torch 
import torchvision
import torch.nn.functional as F
import torchvision.transforms as Transforms


class GPGenerator(object):
    def __init__(self, batch_size, num_classes, data_source, is_train):
        super(GPGenerator, self).__init__()
        self.batch_size = batch_size
        self.num_classes = num_classes 
        
    def generate_mixture_batch(self, is_test):
        dim_input = 1
        dim_output = 1
        batch_size = self.batch_size
        update_batch_size = 8#np.random.randint(5, 11)
        update_batch_size_eval = 30 - update_batch_size
        num_samples_per_class = update_batch_size + update_batch_size_eval
        
        l = 0.5#np.random.uniform(0.1, 0.6, size=batch_size)
        sigma = 1.0#np.random.uniform(0.1, 1.0, size=batch_size)
        
        sel_set = np.zeros(batch_size)

        if is_test:
            init_inputs = np.zeros([batch_size, 1000, dim_input])
            outputs = np.zeros([batch_size, 1000, dim_output])
        else:
            init_inputs = np.zeros([batch_size, num_samples_per_class, dim_input])
            outputs = np.zeros([batch_size, num_samples_per_class, dim_output])

        for func in range(batch_size):
            if is_test:
                init_inputs[func] = np.expand_dims(np.linspace(-2.0, 2.0, num=1000), axis=1)
            else:
                init_inputs[func] = np.random.uniform(-2.0, 2.0,
                                                      size=(num_samples_per_class, dim_input))
                
            x1 = np.expand_dims(init_inputs[func], axis=0)
            x2 = np.expand_dims(init_inputs[func], axis=1)
            
            kernel = sigma**2 * np.exp(-0.5 * np.square(x1-x2) / l**2)
            kernel = np.sum(kernel, axis=-1)
            kernel += (0.02 ** 2) * np.identity(init_inputs[func].shape[0])
            cholesky = np.linalg.cholesky(kernel)
            outputs[func] = np.matmul(cholesky, np.random.normal(size=(init_inputs[func].shape[0], dim_output)))
            
        rng = np.random.default_rng()
        random_idx = rng.permutation(init_inputs.shape[1])
        
        Cx = init_inputs[:, random_idx[:update_batch_size], :]
        Cy = outputs[:, random_idx[:update_batch_size], :]
        Tx = init_inputs
        Ty = outputs
        
        funcs_params = {'l': l, 'sigma': sigma}
        return (Cx, Tx), (Cy, Ty), funcs_params, sel_set

    
class ImageCompletionData(object):
    def __init__(self, batch_size, data_source):
        super(ImageCompletionData, self).__init__()
        self.batch_size = batch_size
        self.data_source = data_source
        
    def loader_generator(self, is_train, is_test):  
        transform = Transforms.Compose([Transforms.Resize(size=(32, 32)), Transforms.ToTensor()])
        
        if self.data_source == 'MNIST':
            dataset = torchvision.datasets.MNIST(root='../data/MNIST', train=is_train,
                                                   download=True, transform=Transforms.ToTensor())
            loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=is_train,
                                                 num_workers=8, drop_last=True)

        elif self.data_source == 'CIFAR10':
            dataset = torchvision.datasets.CIFAR10(root='../data/CIFAR10', train=is_train,
                                                   download=True, transform=transform)
            loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=is_train,
                                                 num_workers=8, drop_last=True)

        elif self.data_source == 'CelebA':
            dataset = torchvision.datasets.ImageFolder(root='../data/CelebA', transform=transform)
            if is_train:
                data = torchvision.datasets.ImageFolder(root='../data/CelebA', transform=transform)
                data.samples = dataset.samples[:162770]
                data.targets = dataset.targets[:162770]
                loader = torch.utils.data.DataLoader(data, batch_size=self.batch_size, shuffle=True, num_workers=8, drop_last=True)
            else:
                data = torchvision.datasets.ImageFolder(root='../data/CelebA', transform=transform)
                if is_test:
                    data.samples = dataset.samples[182637:202599]
                    data.targets = dataset.targets[182637:202599]
                else:
                    data.samples = dataset.samples[162770:182637]
                    data.targets = dataset.targets[162770:182637]
                loader = torch.utils.data.DataLoader(data, batch_size=self.batch_size, shuffle=False, num_workers=8, drop_last=True)
        
        return loader
        
        
