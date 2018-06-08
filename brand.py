from __future__ import print_function, absolute_import

import os
import numpy as np
import random
import math
import pandas as pd
from PIL import Image
import scipy.misc
import skimage.transform
import matplotlib.pyplot as plt

import torch
import torch.utils.data as data
from torchvision import transforms



class Brand(data.Dataset):
    def __init__(self, txtfile, img_prefix, num_classes=100, inp_pre = 256, inp_res=224, train=True, val=False,
                 scale_factor=0.25, rot_factor=20):
        self.img_prefix = img_prefix
        self.num_classes = num_classes
        self.is_train = train           # training set or test set
        self.is_val = val
        self.inp_pre = inp_pre
        self.inp_res = inp_res
        self.scale_factor = scale_factor
        self.rot_factor = rot_factor
        self.anno = self.__genAnnoList__(txtfile)
        self.mean, self.std = self._compute_mean()


    def __genAnnoList__(self,txtfile):
        anno = []
        for row in open(txtfile):
                single = {}
                if self.is_train or self.is_val:
                    single['path'],single['label'] = row[0:len(row)-1].split(' ')
                    single['path'] = self.img_prefix + single['path']
                    single['label'] = int(single['label'])
                else:
                    single['path'] = self.img_prefix + row[0:len(row)-1]
                    single['label'] = 0
                anno.append(single)
        return anno

        # self.mean, self.std = self._compute_mean()

    def _compute_mean(self):
        meanstd_file = 'data/FULL.mean.std.tar'
        if os.path.isfile(meanstd_file):
            meanstd = torch.load(meanstd_file)
        else:
            print('==> compute mean')
            mean = torch.zeros(3)
            std = torch.zeros(3)
            cnt = 0
            for index in range(len(self.anno)):
                cnt += 1
                print( '{} | {}'.format(cnt, len(self.anno)))
                sample = self.anno[index]
                # img_path = os.path.join(self.img_folder, a['img_paths'])
                img = Image.open(sample['path'])
                img = transforms.ToTensor()(img) # CxHxW
                mean += img.view(img.size(0), -1).mean(1)
                std += img.view(img.size(0), -1).std(1)
            mean /= len(self.anno)
            std /= len(self.anno)
            meanstd = {
                'mean': mean,
                'std': std,
                }
            torch.save(meanstd, meanstd_file)
        if self.is_train:
            print('    Mean: %.4f, %.4f, %.4f' % (meanstd['mean'][0], meanstd['mean'][1], meanstd['mean'][2]))
            print('    Std:  %.4f, %.4f, %.4f' % (meanstd['std'][0], meanstd['std'][1], meanstd['std'][2]))
            
        return meanstd['mean'], meanstd['std']

    def scale_holding_resize(self,img,targetsize):
        localimage = img
        scale = min(np.array(targetsize)/np.array([localimage.shape[0],localimage.shape[1]]))
        localimage = skimage.transform.rescale(localimage,scale)
        crop_localimage = np.zeros([targetsize[0],targetsize[1],3])
        crop_localimage[:,:,0] += np.average(localimage[:,:,0])
        crop_localimage[:,:,1] += np.average(localimage[:,:,1])
        crop_localimage[:,:,2] += np.average(localimage[:,:,2])

        height = localimage.shape[0]
        width = localimage.shape[1]
        start_x = 0; start_y = 0;

        if height > width:
            start_x = int((targetsize[1]-width)/2)
            crop_localimage[0:targetsize[0],start_x:start_x+width] = localimage
        else:
            start_y = int((targetsize[0]-height)/2)
            crop_localimage[start_y:start_y+height,0:targetsize[1]] = localimage
        return crop_localimage

    def __getitem__(self, index):
        inp = self.inp_pre
        inr = self.inp_res
        rf = self.rot_factor
        sample = self.anno[index]

        #import pylab
        #fig = plt.figure()
        #ax = fig.add_subplot(121)
        #img = Image.open(sample['path'])
        #img = img.resize((inp,inp))
        #ax.imshow(img)
        img = scipy.misc.imread(sample['path'], mode='RGB')
        img = self.scale_holding_resize(img,[inp,inp])
        img = Image.fromarray(np.uint8(img*255))
        #ax = fig.add_subplot(122)
        #ax.imshow(img)
        #pylab.show()
        #raise 'test break!'

        if self.is_train and not self.is_val:

            # Crop
            crop = transforms.RandomCrop(inr)
            img = crop(img)

            # Flip
            flip = transforms.RandomHorizontalFlip()
            img = flip(img)

            # Color
            # img[0, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
            # img[1, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
            # img[2, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)

            # Rotate
            rotate = transforms.RandomRotation(rf)
            img = rotate(img)
            
        elif self.is_val:
            img = img.resize((inr,inr))
        else:
            # Crop
            crop = transforms.RandomCrop(inr)
            img = crop(img)

            # Flip
            flip = transforms.RandomHorizontalFlip()
            img = flip(img)

            # img = img.resize((inr,inr))
        if self.is_train or self.is_val:
            target = torch.zeros(self.num_classes).long()
            target[sample['label']-1] = 1
            
        else:
            target = torch.IntTensor(1,1).zero_()
        # Prepare image and groundtruth map
        # inp = crop(img, c, s, [self.inp_res, self.inp_res], rot=r)
        inp = transforms.ToTensor()(img)
        inp = transforms.Normalize(self.mean, self.std)(inp)

        # Meta info
        # if is_octopus:
        #     meta = {'index' : index, 
        #     'img_path' : sample['path'],
        #     'type' : sample['type'],
        #     'attribute' : sample['attribute']}
        # else:
        meta = {'index' : index, 
            'img_path' : sample['path'],
            'label' : sample['label']}

        return inp, target, meta

    def __len__(self):
        return len(self.anno)

    def __get_numlabel__(self):
        return len(self.anno[0]['attribute'])

