import sys
sys.path.append('C:/Users/111/Desktop/standford_dog_classification')

import os
import random
from config import config
from tqdm import tqdm   # 进度条

if not os.path.exists('./data'):   # 判断、创建data文件夹
    os.makedirs('./data')
    
train_txt = open('./data/train.txt', 'w')          # data文件夹下创建train_txt.txt
val_txt = open('./data/valid.txt', 'w')            # data文件夹下创建val_txt.txt
label_txt = open('./data/label_list.txt', 'w')     # data文件夹下创建label_txt.txt


label_list = []

for dir in tqdm(os.listdir(config.data_root)):      # dir遍历data_root目录下的image的图片文件名
    if dir not in label_list:
        label_list.append(dir)                      # 将image下的图片名填入label_list
        label_txt.write('{} {}\n'.format(dir, str(len(label_list)-1)))   # dir:写入内容为图片名，str(len(label_list)-1))图片名对应序号
        data_path = os.path.join(config.data_root, dir)  # 在data_root路径下拼接一个dir路径，并用data_path指向
        train_list = random.sample(os.listdir(data_path),  # random.sample('xxx',n)在‘xxx'中取n个
                                   int(len(os.listdir(data_path))*0.8))
        for im in train_list:
            train_txt.write('{}/{} {}\n'.format(dir, im, str(len(label_list)-1)))  # dir:写入内容为图片名/图片名+序号，如mobilehomepark/mobilehomepark01.tif 0
        for im in os.listdir(data_path):
            if im in train_list:
                continue
            else:
                val_txt.write('{}/{} {}\n'.format(dir, im, str(len(label_list)-1))) # val_txt写入剩下的20%图片名/图片名+序号，如mobilehomepark/mobilehomepark01.tif 0

# 写入test.txt
test_txt = open('./data/test.txt','w')
test_list = []
for dir in tqdm(os.listdir(config.test_root)):      # dir遍历data_root目录下的image的图片文件名
    if dir not in test_list:
        test_list.append(dir)
        test_txt.write('{} {}\n'.format(dir,len(test_list)-1))

