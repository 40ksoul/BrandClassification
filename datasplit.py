from __future__ import print_function, absolute_import

import os
import numpy as np
import random
import math
import pandas as pd
from PIL import Image
import scipy.misc
import skimage.transform

import torch
import torch.utils.data as data
from torchvision import transforms

def split():
    train_file = 'data/train.txt'
    trainb_file = open('data/trainb.txt','w')
    val_file = open('data/val.txt','w')
    count = np.zeros(100)
    for row in open(train_file):
        r = random.random()
        _ , label = row[0:len(row)-1].split(' ')
        if r > 0.5 and count[int(label)-1] <3:
            count[int(label)-1] += 1
            val_file.write(row)
        else:
            trainb_file.write(row)

def state():
    count = np.zeros(100)
    for row in open('data/train.txt'):
        _ , label = row[0:len(row)-1].split(' ')
        count[int(label)-1] += 1
    print(count)
    count = np.zeros(100)
    for row in open('data/trainb.txt'):
        _ , label = row[0:len(row)-1].split(' ')
        count[int(label)-1] += 1
    print(count)
    count = np.zeros(100)
    for row in open('data/val.txt'):
        _ , label = row[0:len(row)-1].split(' ')
        count[int(label)-1] += 1
    print(count)

def compare():
    result1 = []
    result2 = []
    comparefile = open('compare_senet_inner_new20a5vsold20.txt','w')
    for row in open('results/result_512_senet154_aug7_epoch20.txt'):
        single = {}
        single['path'],single['label'] = row[0:len(row)-1].split(' ')
        result1.append(single)
    for row in open('results/result_512_senet154_aug7.txt'):
        single = {}
        single['path'],single['label'] = row[0:len(row)-1].split(' ')
        result2.append(single)
    for i in range(1000):
        if result1[i]['label'] != result2[i]['label']:
            comparefile.write(result1[i]['path']+' '+result1[i]['label']+' '+result2[i]['label']+'\n')


if __name__ == '__main__':
    #compare()
    split()


