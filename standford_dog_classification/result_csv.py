# -*- coding:utf-8 -*-
from dataset_test import read_txt,RSDataset
import csv
import os
import torch
from config import config
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

def inference():
    model = torch.load(os.path.join('./checkpoints', config.checkpoint))
    model
    transform = transforms.Compose([transforms.Resize((config.width, config.height)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])])
    dst_valid = RSDataset('./data/test.txt', width=config.width,
                          height=config.height, transform=transform)

    dataloader_test = DataLoader(dst_valid, shuffle=False, batch_size= 1,      # batch size = 1 for test
                                  num_workers=config.num_workers)
    pred_categlory = []
    for ims,_ in dataloader_test:
        input = Variable(ims)
        output = model(input)
        _, pred = output.topk(1, 1, True, True)    # 根据batch size = 1, 返回一个预测的tensor, 如，tensor([2])
        print('pred_tensor ', pred)

        pred_cate = pred.numpy().item()          # tensor to numpy to num,如tensor([2])->[2]->2
        print('pred_numpy ', pred_cate)
        pred_categlory.append(pred_cate)
    print('cataglory ',pred_categlory)
    return pred_categlory

def creat_csv_with_index(pred_categlory):
    pred_id,_ = read_txt('./data/test.txt')
    print(pred_id)
    if not os.path.exists('./data/test.csv'):
        with open('./data/test.csv','a+',newline='') as f:
            csv_write = csv.writer(f)
            csv_write.writerow(['id','category'])

    with open('./data/test.csv', 'a+', newline='\n') as f:
        csv_write = csv.writer(f)
        for i in range(len(pred_id)):
            csv_write.writerow([pred_id[i], pred_categlory[i]])


if __name__ == '__main__':
    creat_csv_with_index(pred_categlory=inference())
