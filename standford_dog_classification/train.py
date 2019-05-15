# -*- coding:utf-8 -*-
import sys
sys.path.append('C:/Users/111/Desktop/A/Remote-Sensing-Image-Classification-master')
import time

from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
from dataset import *
from networks.network import *
from lr_schedule import *
from metric import *
# from utils.plot import *
from config import config


def train():
    # model
    if config.model == 'ResNet18':
        backbone = models.resnet18(pretrained=True)
        model = ResNet18(backbone, num_classes=config.num_classes)
    elif config.model == 'ResNet34':
        backbone = models.resnet34(pretrained=True)
        model = ResNet34(backbone, num_classes=config.num_classes)
    elif config.model == 'ResNet50':
        backbone = models.resnet50(pretrained=True)
        model = ResNet50(backbone, num_classes=config.num_classes)
    elif config.model == 'ResNet101':
        backbone = models.resnet101(pretrained=True)
        model = ResNet101(backbone, num_classes=config.num_classes)
    elif config.model == 'ResNet152':
        backbone = models.resnet152(pretrained=True)
        model = ResNet152(backbone, num_classes=config.num_classes)
    else:
        print('ERROR: No model {}!!!'.format(config.model))
    # print(model)
    # model = torch.nn.DataParallel(model)
    model

    # freeze layers
    if config.freeze:
        for p in model.backbone.layer1.parameters(): p.requires_grad = False  # requires_grad = Fasle 时不需要更新梯度， 适用于冻结某些层的梯度
        for p in model.backbone.layer2.parameters(): p.requires_grad = False
        for p in model.backbone.layer3.parameters(): p.requires_grad = False
        # for p in model.backbone.layer4.parameters(): p.requires_grad = False

    # loss
    criterion = nn.CrossEntropyLoss()

    # train data
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.RandomResizedCrop(224),  # 随机剪裁
                                    transforms.RandomHorizontalFlip(),  # 依概率p水平翻转，默认p=0.5
                                    transforms.ColorJitter(0.05, 0.05, 0.05),
                                    # 修改亮度、对比度和饱和度class torchvision.transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)
                                    transforms.RandomRotation(10),  # 随机旋转
                                    transforms.Resize((config.width, config.height)),  # resize
                                    transforms.ToTensor(),  # 转为tensor，并归一化至[0-1]
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])])  # 标准化
    dst_train = RSDataset('./data/train.txt', width=config.width,
                          height=config.height, transform=transform)

    dataloader_train = DataLoader(dst_train, shuffle=True, batch_size=int(config.batch_size),
                                  num_workers=config.num_workers)

    # validation data
    transform = transforms.Compose([transforms.Resize((config.width, config.height)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])])
    dst_valid = RSDataset('./data/valid.txt', width=config.width,
                          height=config.height, transform=transform)
    dataloader_valid = DataLoader(dst_valid, shuffle=False, batch_size=int(config.batch_size / 2),
                                  num_workers=config.num_workers)

    # log
    if not os.path.exists('./log'):
        os.makedirs('./log')
    log = open('./log/log.txt', 'a')

    log.write('-' * 30 + '\n')
    log.write(
        'model:{}\nnum_classes:{}\nnum_epoch:{}\nlearning_rate:{}\nim_width:{}\nim_height:{}\niter_smooth:{}\n'.format(
            config.model, config.num_classes, config.num_epochs, config.lr,
            config.width, config.height, config.iter_smooth))

    # load checkpoint
    if config.resume:
        model = torch.load(os.path.join('./checkpoints', config.checkpoint))

    # train
    sum = 0
    train_loss_sum = 0
    train_top1_sum = 0
    # train_top5_sum = 0
    max_val_acc = 0
    train_draw_acc = []
    val_draw_acc = []
    loss_plot = []
    for epoch in range(config.num_epochs):  # num_epochs = 100
        ep_start = time.time()

        # adjust lr
        # lr = half_lr(config.lr, epoch)
        lr = step_lr(epoch)

        # optimizer
        # optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.0002)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                     lr=lr, betas=(0.9, 0.999), weight_decay=0.0002)

        model.train()
        top1_sum = 0
        for i, (ims, label) in enumerate(dataloader_train):
            input = Variable(ims)
            target = Variable(label).long()  # long() 函数将数字或字符串转换为一个长整型。

            output = model(input)
            print('out_put ',output)
            loss = criterion(output, target)
            # for plotting loss
            loss_plot.append(loss)

            optimizer.zero_grad()  # 将模型参数梯度设为0
            loss.backward()  # 误差反向求导
            optimizer.step()   # 参数更新

            top1 = accuracy(output.data, target.data, topk=(1,))  # 根据训练结果output和标签target，求TOP1或者TOP5--topk=(1,5)
            print('top1 ',top1)
            # top5 = accuracy(output.data, target.data, topk=(1,5))

            train_loss_sum += loss.data.cpu().numpy()       # 把数据loss放在CPU上,并转化成numpy格式
            train_top1_sum += top1[0]
            # train_top5_sum += top5[0]
            sum += 1
            top1_sum += top1[0]    # top1每次只存一个数top1  [tensor([62.5000])]，tensor 形式

            if (i + 1) % config.iter_smooth == 0:
                print('Epoch [%d/%d], Iter [%d/%d], lr: %f, Loss: %.4f, top1: %.4f'
                      % (epoch + 1, config.num_epochs, i + 1, len(dst_train) // config.batch_size,
                         lr, train_loss_sum / sum, train_top1_sum / sum))
                log.write('Epoch [%d/%d], Iter [%d/%d], lr: %f, Loss: %.4f, top1: %.4f\n'
                          % (epoch + 1, config.num_epochs, i + 1, len(dst_train) // config.batch_size,
                             lr, train_loss_sum / sum, train_top1_sum / sum))
                sum = 0
                train_loss_sum = 0
                train_top1_sum = 0

        train_draw_acc.append(top1_sum / len(dataloader_train))

        epoch_time = (time.time() - ep_start) / 60.

        if epoch % 1 == 0 and epoch < config.num_epochs:
            # eval
            val_time_start = time.time()
            val_loss, val_top1 = eval(model, dataloader_valid, criterion)
            val_draw_acc.append(val_top1)
            val_time = (time.time() - val_time_start) / 60.

            print('Epoch [%d/%d], Val_Loss: %.4f, Val_top1: %.4f, val_time: %.4f s'
                  % (epoch + 1, config.num_epochs, val_loss, val_top1, val_time * 60))
            print('epoch time: {}s'.format(epoch_time * 60))
            if val_top1[0].data > max_val_acc:
                max_val_acc = val_top1[0].data
                print('Taking snapshot...')
                if not os.path.exists('./checkpoints'):
                    os.makedirs('./checkpoints')
                torch.save(model, '{}/{}.pth'.format('checkpoints', config.model))

            log.write('Epoch [%d/%d], Val_Loss: %.4f, Val_top1: %.4f, val_time: %.4f s\n'
                      % (epoch + 1, config.num_epochs, val_loss, val_top1, val_time * 60))
        # ########################## draw_curve(train_draw_acc, val_draw_acc)
    log.write('-' * 30 + '\n')
    log.close()
    # plot loss
    y = loss_plot
    x = range(0, len(y), 1)
    plt.plot(x, y)


# validation
def eval(model, dataloader_valid, criterion):
    sum = 0
    val_loss_sum = 0
    val_top1_sum = 0
    model.eval()
    for ims, label in dataloader_valid:
        input_val = Variable(ims)
        target_val = Variable(label)
        output_val = model(input_val)
        loss = criterion(output_val, target_val)
        # probs = F.softmax(output_val)
        # print(probs)
        top1_val = accuracy(output_val.data, target_val.data, topk=(1,))

        sum += 1
        val_loss_sum += loss.data.cpu().numpy()
        val_top1_sum += top1_val[0]
    avg_loss = val_loss_sum / sum
    avg_top1 = val_top1_sum / sum
    return avg_loss, avg_top1


if __name__ == '__main__':
    train()


