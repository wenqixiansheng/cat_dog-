#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/16 9:40
# @Author  : QiJun
# @File    : data_input.py
# @算法思想 :  数据处理，标准化


import os
import numpy as np
import cv2

# 数据集处理

"""
    一、生成图片路径和标签的List
"""

# step1获取data下所有的猫图路径名，存放到cats中，同时贴上标签0，存放到label_cats中。狗图同理。


cats = []
label_cats = []
dogs = []
label_dogs = []
# 对于已经分好文件夹的数据集
def get_files2(file_dir):
    for file in os.listdir(file_dir+'cat'):
            cats.append(file_dir +'cat/'+ file)
            label_cats.append(0)  # 标签
    for file in os.listdir(file_dir+'dog'):
            dogs.append(file_dir +'dog/'+file)
            label_dogs.append(1)

    # step2：对生成的图片路径和标签List做打乱处理
    #把cat和dog合起来组成一个list（img和lab）
    image_list = np.hstack((cats, dogs))
    label_list = np.hstack((label_cats, label_dogs))

    #利用shuffle打乱顺序
    temp = np.array([image_list, label_list])
    print(temp.shape)
    temp = temp.transpose()
    np.random.shuffle(temp)

    #从打乱的temp中再取出list（img和lab）
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]


# 针对没有分好文件夹的数据
def get_files(file_dir):
    for file in os.listdir(file_dir):
        name = file.split(sep='.')
        if name[0] == 'cat':
            img = cv2.imread(file_dir + file, 1)
            img = cv2.resize(img, (32, 32))
            cats.append(img)
            label_cats.append(0)
        else:
            img = cv2.imread(file_dir + file, 1)
            img = cv2.resize(img, (32, 32))
            dogs.append(img)
            label_dogs.append(1)
    print("There are %d cats\nThere are %d dogs" % (len(cats), len(dogs)))

    image_list = cats + dogs
    label_list = label_cats + label_dogs

    image = np.array(image_list)  # 将list转化为array
    label = np.array(label_list)
    label.resize(len(label),1)
    print("y0",label.shape,label[0])
    return image,label
