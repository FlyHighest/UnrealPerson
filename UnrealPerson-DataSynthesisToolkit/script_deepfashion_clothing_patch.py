#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tues 1st Sep 2020

@author: zhangtianyu
"""

import os
import glob
import scipy.io 
import tqdm
import PIL.Image
import numpy as np
from multiprocessing import cpu_count
from multiprocessing import Pool

def mt_func(seg,pixel_label):
    print(seg)
    gender = "m" if seg.find('WOMEN')==-1 else "f"
    idx = seg.split('/')[-2]+'_'+seg.split('/')[-1][:4]
    img_path = seg.replace('_segment.png','.jpg')
    if not os.path.exists(img_path):
        print(img_path+" does not exist.")
        return False
    

    img = PIL.Image.open(img_path)
    gt = PIL.Image.open(seg)
    gt = np.array(gt)
    gt = gt[:,:,:3]
    top1=0
    top1_count=0
    top2=0
    top2_count=0
    for i in pixel_label.keys():
        
        count=match_color(gt,i).sum()
        if count==0:continue
        if count>top1_count:
            top2_count=top1_count
            top2=top1
            top1_count=count
            top1=i
        elif count>top2_count:
            top2_count=count
            top2=i
    
    top1_gt = match_color(gt,top1)
    top2_gt = match_color(gt,top2)
        
    
    
    a_1 = findMaxRect(top1_gt)
    cood=a_1[1][0]
    rec_1=img.crop((cood[1],cood[0],cood[3],cood[2]))
    
    a_2 = findMaxRect(top2_gt)
    cood=a_2[1][0]
    rec_2=img.crop((cood[1],cood[0],cood[3],cood[2]))
    
    rec_1.save(save_dir+"/"+pixel_label[top1]+"_{}_{}.png".format(gender,idx))
    rec_2.save(save_dir+"/"+pixel_label[top2]+"_{}_{}.png".format(gender,idx))
    return True
    
    

def match_color(mask, color):
    match_region = np.ones(mask.shape[:2],dtype=bool)
    for c in range(3):
        val = color[c]
        channel_region= mask[:,:,c]==val
        match_region &= channel_region
    return match_region

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

save_dir = '/Users/zhangtianyu/Downloads/clothing-co-parsing-master/df_patches'
img_dir = '/Users/zhangtianyu/Downloads/df_clothes/img_highres_seg'


pixel_label = { 
                (255,250,250)   : "top",
                (250,235,215)   : "skirt",
                (255, 250, 205) : "dress",
                (220, 220, 220) : "outer",
                (211, 211, 211) : "pants",
                (127, 255, 212) : "headwear",          
              }


seg_imgs = glob.glob(os.path.join(img_dir,"*/*/*/*_segment.png"))
seg_imgs.sort()
results=[]
mt_pool = Pool(cpu_count())
for seg in seg_imgs:
    params=(seg,pixel_label)
    results.append(mt_pool.apply_async(mt_func,params))
    
mt_pool.close()
mt_pool.join()


