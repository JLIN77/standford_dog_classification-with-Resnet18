# -*- coding:utf-8 -*-
import sys
sys.path.append('C:/Users/111/Desktop/standford_dog_classification')  # sys.path.append()添加路径,用于调用模块
import os
import cv2
import pandas as pd
import numpy as np
from PIL import Image, ImageFilter

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

from config import config

def read_txt(path):   # 获取txt文件图片名和标签（在txt文件下）
    ims, labels = [], []
    with open(path, 'r') as f:
        for line in f.readlines():
            im, label = line.strip().split(' ')  # n02085620-Chihuahua 0  删除行末换行，以‘  ’分开
            ims.append(im)
            labels.append(int(label))
    return ims, labels


class RSDataset(Dataset):   # 返回真实的图片、标签、图片数量
    def __init__(self, txt_path, width=256, height=256, transform=None, test=False):
        self.ims, self.labels = read_txt(txt_path)  # 读图片名、读标签
        self.width = width
        self.height = height
        self.transform = transform
        self.test = test

    def __getitem__(self, index):
        im_path = self.ims[index]
        label = self.labels[index]
        im_path = os.path.join(config.test_root, im_path)   # im_path指向data_root路径下某个具体的图片
        im = Image.open(im_path)  # Image方式读取image
        # im = im.resize((self.width, self.height))
        if self.transform is not None:
            im = self.transform(im)  # 用transforms.ToTensor(),将numpy的ndarray或PIL.Image读的图片转换成形状为(C,H, W)的Tensor格式，且/255归一化到[0,1.0]之间

        return im, label

    def __len__(self):
        return len(self.ims)


if __name__ == '__main__':
    n=0
    if not os.path.exists('tensor_txt'):
        tensor_txt = open('./data/tensor.txt', 'w')
    transform = transforms.Compose([transforms.ToTensor()])
    dst_train = RSDataset('./data/train.txt', width=256, height=256, transform=transform)
    dataloader_train = DataLoader(dst_train, shuffle=True, batch_size=1, num_workers=0)
    # pytorch Dataloader(dataset, batch_size=1, shuffle=False, sampler=None, num_workers=0, collate_fn=default_collate, pin_memory=False, drop_last=False)
    # dataset加载对象，batch_size=batch size,shuffle是否打乱数据,sampler样本抽样，num_workers:使用多进程加载的进程数，0代表不使用多进程,
    # collate_fn： 如何将多个样本数据拼接成一个batch，一般使用默认的拼接方式即可,pin_memory：是否将数据保存在pin memory区，pin memory中的数据转到GPU会快一些
    # drop_last：dataset中的数据个数可能不是batch_size的整数倍，drop_last为True会将多出来不足一个batch的数据丢弃

    # for im, loc, cls in dataloader_train:

    for data in dataloader_train:

        print(data)
        n+=1
        print(n)
        # print loc, cls

        tensor_txt.write('{}'.format(data))
