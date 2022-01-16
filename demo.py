#!/usr/bin/env python
# coding: utf-8

import os
import time
import argparse
import torch as t
from torch.utils.data import DataLoader
import numpy as np

from utils import get_train_test

os.environ['CUDA_VISIBLE_DEVICES'] = '2'


def norm_ip(img, min, max):
    temp = t.clamp(img, min=min, max=max)
    temp = (temp + -min) / (max - min + 1e-5)
    return temp


device = t.device('cuda' if t.cuda.is_available() else 'cpu')


args = argparse.Namespace()
args.dataset = 'cifar10'
args.data_root = '../../data'
args.seed = 1
data_loader, test_loader = get_train_test(args)

# In[9]:


## Load real data

from data import Cifar10

test_dataset = Cifar10(args, full=True, noise=False)
test_dataloader = DataLoader(test_dataset, batch_size=100, num_workers=0, shuffle=True, drop_last=False)
test_ims = []


def rescale_im(im):
    return np.clip(im * 256, 0, 255).astype(np.uint8)


for data_corrupt, data, label_gt in test_dataloader:
    if args.dataset in ['celeba128', 'img32']:
        data = data_corrupt.numpy().transpose(0, 2, 3, 1)
    else:
        data = data.numpy()
    test_ims.extend(list(rescale_im(data)))
    if (args.dataset == "imagenet" or 'img' in args.dataset) and len(test_ims) > 60000:
        test_ims = test_ims[:60000]
        break

print(np.array(test_ims).shape)


## read different datasets to evaluate their IS/FID
# training cifar10 dataset again
train_images = []
for data, y, _ in data_loader:
    data = norm_ip(data, -1, 1).numpy().transpose(0, 2, 3, 1)
    train_images.extend(list(rescale_im(data)))
    if (args.dataset == "imagenet" or 'img' in args.dataset) and len(train_images) > 60000:
        train_images = train_images[:60000]
        break

# testing cifar10 dataset
test_images = []
for data, y, _ in test_loader:
    data = norm_ip(data, -1, 1).numpy().transpose(0, 2, 3, 1)
    test_images.extend(list(rescale_im(data)))
    if (args.dataset == "imagenet" or 'img' in args.dataset) and len(test_images) > 60000:
        test_images = test_images[:60000]
        break


# call the function of TF
from inception import get_inception_score
from fid import get_fid_score

# benchmarks

# cifar10 train set, test set,  train_set[:10000]

start = time.time()
fid = get_fid_score(test_images, test_ims)
print("FID of score {} takes {}s".format(fid, time.time() - start))

# In[82]:


start = time.time()
fid = get_fid_score(train_images, test_ims)
print("FID of score {} takes {}s".format(fid, time.time() - start))

# In[81]:


start = time.time()
fid = get_fid_score(np.array(train_images)[:10000], test_ims)
print("FID of score {} takes {}s".format(fid, time.time() - start))

# In[80]:


splits = max(1, len(test_images) // 5000)
start = time.time()
score, std = get_inception_score(test_images, splits=splits)
print("Inception score of {} with std of {} takes {}s".format(score, std, time.time() - start))

## A generated dataset


feed_imgs = np.load('feed_imgs.npy')


start = time.time()
fid = get_fid_score(feed_imgs, test_ims)
print("FID of score {} takes {}s".format(fid, time.time() - start))


splits = max(1, len(feed_imgs) // 5000)
start = time.time()
score, std = get_inception_score(feed_imgs, splits=splits)
print("Inception score of {} with std of {} takes {}s".format(score, std, time.time() - start))
