import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from torchvision import transforms

from .rerank import re_ranking
from .ranking import cmc, mean_ap
from .st_distribution import joint_similarity
from .util import get_info, l2_dist

def compute_joint_dist(distribution, q_feas, g_feas, q_frames, g_frames, q_cams, g_cams):
    dists = []
    for i in range(len(q_frames)):
        dist = joint_similarity(
            q_feas[i],q_cams[i],q_frames[i], 
            g_feas,   g_cams,   g_frames,
            distribution)
        
        dist = np.expand_dims(dist, axis=0)
        dists.append(dist)
        #print(i, dist.shape)
    dists = np.concatenate(dists, axis=0)

    return dists

def evaluate_joint(test_fea, st_distribute, ann_file, select_set='duke'):
    #fea_duke_test = np.load('duke_test_feas.npy')
    print( 'test feature', test_fea.shape)
    #print(len(cams))
    #2228 for duke, 3368 for market, 11659 for msmt17
    if select_set == 'duke':
        query_num = 2228 
    elif select_set == 'market':
        query_num = 3368
    
    if type(ann_file)==tuple:
        labels,cams,frames,is_query = ann_file
    
        query_labels = labels[is_query]
        query_cams = cams[is_query]
        query_frames = frames[is_query]
        query_features = test_fea[is_query]

        gallery_labels = labels[~is_query]
        gallery_cams = cams[~is_query]
        gallery_frames = frames[~is_query]    
        gallery_features = test_fea[~is_query]

    else:
        labels, cams, frames = get_info(ann_file)
        query_labels = labels[0:query_num]
        query_cams = cams[0:query_num]
        query_frames = frames[0:query_num]
        query_features = test_fea[0:query_num, :]

        gallery_labels = labels[query_num:]
        gallery_cams = cams[query_num:]
        gallery_frames = frames[query_num:] 
        gallery_features = test_fea[query_num:, :]


    dist = l2_dist(query_features, gallery_features)
    mAP = mean_ap(dist, query_labels, gallery_labels, query_cams, gallery_cams)
    cmc_scores = cmc(dist, query_labels, gallery_labels, query_cams, gallery_cams,
        separate_camera_set=False, single_gallery_shot=False, first_match_break=True)
    print('performance based on visual similarity')
    print('mAP: %.4f, r1:%.4f, r5:%.4f, r10:%.4f, r20:%.4f'%(mAP, cmc_scores[0], cmc_scores[4], cmc_scores[9], cmc_scores[19]))
    if st_distribute is None:
        return 

    #st_distribute = np.load('distribution_duke_train.npy')
    dist = compute_joint_dist(st_distribute, 
        query_features, gallery_features, 
        query_frames, gallery_frames, 
        query_cams, gallery_cams)

    mAP = mean_ap(dist, query_labels, gallery_labels, query_cams, gallery_cams)
    cmc_scores = cmc(dist, query_labels, gallery_labels, query_cams, gallery_cams,
        separate_camera_set=False, single_gallery_shot=False, first_match_break=True)
    print('performance based on joint similarity')
    print('mAP: %.4f, r1:%.4f, r5:%.4f, r10:%.4f, r20:%.4f'%(mAP, cmc_scores[0], cmc_scores[4], cmc_scores[9], cmc_scores[19]))
