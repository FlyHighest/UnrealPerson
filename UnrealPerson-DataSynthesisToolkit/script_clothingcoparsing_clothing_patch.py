#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 15:07:13 2020

@author: zhangtianyu
"""

import os
import glob
import scipy.io 
import tqdm
import PIL.Image
import numpy as np

def findMaxRect(data):
   
    '''http://stackoverflow.com/a/30418912/5008845'''

    nrows,ncols = data.shape
    w = np.zeros(dtype=int, shape=data.shape)
    h = np.zeros(dtype=int, shape=data.shape)
    skip = 0
    area_max = (0, [])
   

    for r in range(nrows):
        for c in range(ncols):
            if data[r][c] == skip:
                continue
            if r == 0:
                h[r][c] = 1
            else:
                h[r][c] = h[r-1][c]+1
            if c == 0:
                w[r][c] = 1
            else:
                w[r][c] = w[r][c-1]+1
            minw = w[r][c]
            for dh in range(h[r][c]):
                minw = min(minw, w[r-dh][c])
                area = (dh+1)*minw
                if area > area_max[0]:
                    area_max = (area, [(r-dh, c-minw+1, r, c)])

    return area_max

save_dir = '/Users/zhangtianyu/Downloads/clothing-co-parsing-master/ccp_patches'
img_dir = '/Users/zhangtianyu/Downloads/clothing-co-parsing-master/photos'
anno_dir =  '/Users/zhangtianyu/Downloads/clothing-co-parsing-master/annotations/pixel-level'
labels = scipy.io.loadmat('/Users/zhangtianyu/Downloads/clothing-co-parsing-master/label_list.mat')
labels= labels['label_list']

pixel_label = {}

for i in range(59):
    pixel_label[i]=str(labels[0][i][0])
    

annos = glob.glob(anno_dir + '/*.mat')


anno_path = annos[0]
for anno_path in tqdm.tqdm(annos):
    num=os.path.split(anno_path)[1].split('.')[0]
    
    img_path = anno_path.replace('annotations/pixel-level','photos').replace('.mat','.jpg')
    mat = scipy.io.loadmat(anno_path)
    gt = mat['groundtruth']
    img = PIL.Image.open(img_path)
    
    top1=0
    top1_count=0
    top2=0
    top2_count=0
    for i in range(1,59):
        if i==41:continue
        count=(gt==i).sum()
        if count==0:continue
        if count>top1_count:
            top2_count=top1_count
            top2=top1
            top1_count=count
            top1=i
        elif count>top2_count:
            top2_count=count
            top2=i
    
    top1_gt = gt==top1
    top2_gt = gt==top2
        
    
    
    a_1 = findMaxRect(top1_gt)
    cood=a_1[1][0]
    rec_1=img.crop((cood[1],cood[0],cood[3],cood[2]))
    
    a_2 = findMaxRect(top2_gt)
    cood=a_2[1][0]
    rec_2=img.crop((cood[1],cood[0],cood[3],cood[2]))
    
    rec_1.save(save_dir+"/"+pixel_label[top1]+"_{}.png".format(num))
    rec_2.save(save_dir+"/"+pixel_label[top2]+"_{}.png".format(num))
