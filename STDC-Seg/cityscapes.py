#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import os.path as osp
import os
from PIL import Image
import numpy as np
import json

from transform import *
from glob import glob


class CityScapes(Dataset):
    def __init__(self, rootpth, cropsize=(640, 480), mode='train', 
    randomscale=(0.125, 0.25, 0.375, 0.5, 0.675, 0.75, 0.875, 1.0, 1.25, 1.5), *args, **kwargs):
        super(CityScapes, self).__init__(*args, **kwargs)
        assert mode in ('train', 'val', 'test', 'trainval')
        self.mode = mode
        self.imgs=[]
        self.labels=[]
        if self.mode=='train':
            file=open("/project/train/src_repo/train.txt",'r')
            lines=file.readlines()
            for line in lines:
                self.imgs.append(line[:-1])
                self.labels.append(line[:-4]+"png")
        elif self.mode=='val':
            file = open("/project/train/src_repo/val.txt", 'r')
            lines = file.readlines()
            for line in lines:
                self.imgs.append(line[:-1])
                self.labels.append(line[:-4]+"png")

        ## pre-processing
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        self.trans_train = Compose([
            ColorJitter(
                brightness = 0.5,
                contrast = 0.5,
                saturation = 0.5),
            HorizontalFlip(),
            # RandomScale((0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)),
            RandomScale(randomscale),
            # RandomScale((0.125, 1)),
            # RandomScale((0.125, 0.25, 0.375, 0.5, 0.675, 0.75, 0.875, 1.0)),
            # RandomScale((0.125, 0.25, 0.375, 0.5, 0.675, 0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5)),
            RandomCrop(cropsize)
            ])


    def __getitem__(self, idx):
        impth = self.imgs[idx]
        lbpth = self.labels[idx]
        img = Image.open(impth).convert('RGB')
        label = Image.open(lbpth)
        # if self.mode == 'train' or self.mode == 'trainval':
        im_lb = dict(im = img, lb = label)
        im_lb = self.trans_train(im_lb)
        img, label = im_lb['im'], im_lb['lb']
        img = self.to_tensor(img)
        label = np.array(label).astype(np.int64)[np.newaxis, :]
        # label = self.convert_labels(label)
        return img, label


    def __len__(self):
        return len(self.imgs)


    def convert_labels(self, label):
        for k, v in self.lb_map.items():
            label[label == k] = v
        return label



if __name__ == "__main__":
    from tqdm import tqdm
    ds = CityScapes('./data/', n_classes=19, mode='val')
    uni = []
    for im, lb in tqdm(ds):
        lb_uni = np.unique(lb).tolist()
        uni.extend(lb_uni)
    print(uni)
    print(set(uni))

