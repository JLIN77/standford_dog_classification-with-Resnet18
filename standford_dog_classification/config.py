# -*- coding:utf-8 -*-
import torch
print(torch.__version__)
class DefaultConfigs(object):
    data_root = 'C:/Users/111/Desktop/B+ Images' # 数据集的根目录
    test_root = 'C:/Users/111/Desktop/B+ Images test'

    model = 'ResNet18' # ResNet18, ResNet34, ResNet50, ResNet101, ResNet152 使用的模型
    freeze = True # 是否冻结卷基层

    seed = 1000 # 固定随机种子

    num_workers = 0 # DataLoader 中的多线程数量  if CPU, =0 .elseif GPU > 0
    num_classes = 3 # 分类类别数
    num_epochs = 2
    batch_size = 16
    lr = 0.01 # 初始lr
    width = 256 # 输入图像的宽
    height = 256 # 输入图像的高
    iter_smooth = 1 # 打印&记录log的频率

    # resume = False #
    resume = True
    checkpoint = 'ResNet18.pth' # 训练完成的模型名

config = DefaultConfigs()