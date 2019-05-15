# standford_dog_classification-with-Resnet18
This is my first deep learning program.

Dataset: It contains two directories, one for train and validation(150 pictures), the other for test(13 pictures)

Configuration: AMD RYZEN5 cpu + pycharm 2018.1.2 + python 3.6 + pytorch 1.0.1  

This project contains several files, which I list as below:

--creat_img_list.py

--config.py

--networks->network

--data->(label_list.txt + train.txt + valid.txt + test.txt + test.cvs)

--dataset.py

--dataset_test.py

--metric.py

--lr_schedule.py

--train.py

--result_to_csv.py

--log->log.txt

ps：In order to avoid unnessary path error, I highly recommend you to add the following 3 codes for each of the .py files.
···python
 -*- coding:utf-8 -*-
 import sys
 sys.path.append('C:/Users/111/Desktop/standford_dog_classification')  # for your project path
···
