from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np
import torch,time
from torch import nn
import torch.nn.functional as F
import random


def pairwise_loss(outputs,  label1, seg,sigmoid_param=1.0,
                  l_threshold=15.0):
    N = label1.size(0)
    similarity = label1.expand(N, N).eq(label1.expand(N, N).t()).float()
    dot_product =  sigmoid_param *torch.mm(outputs[seg], outputs[seg].t())
    print(dot_product)

    exp_product = torch.exp(dot_product)
    
    print(exp_product)
    mask_dot = dot_product.data > l_threshold
    mask_exp = dot_product.data <= l_threshold
    mask_positive = similarity.data > 0
    mask_negative = similarity.data <= 0
    mask_dp = mask_dot & mask_positive
    mask_dn = mask_dot & mask_negative
    mask_ep = mask_exp & mask_positive
    mask_en = mask_exp & mask_negative

    dot_loss = dot_product * (1-similarity)
    exp_loss = (torch.log(1+exp_product) - similarity * dot_product)
    pos_weight = 10
    pos_loss = (torch.sum(torch.masked_select(exp_loss, mask_ep)) +
            torch.sum(torch.masked_select(dot_loss, mask_dp)))
    neg_loss = torch.sum(torch.masked_select(exp_loss, mask_en)) + \
        torch.sum(torch.masked_select(dot_loss, mask_dn))

    pos_loss = pos_loss*pos_weight / torch.sum(mask_positive.float())
    neg_loss = neg_loss / torch.sum(mask_negative.float())
    loss = pos_loss  +  neg_loss

    return loss, pos_loss, neg_loss

def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def cosine_dist(x, y):
    # m, n = x.size(0), y.size(0)
    dist = 1 - torch.matmul(x, y.t())
    # dist = torch.ones(dist.size()).float() - dist
    return dist


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

def euclidean_dist_elementwise(x,y):
    xx=torch.pow(x,2).sum(1)
    yy=torch.pow(y,2).sum(1)
    xy=(x*y).sum(1)
    dist=xx+yy-2*xy
    dist=dist.clamp(min=1e-12).sqrt()
    return dist


def hard_example_mining(dist_mat, labels,kthp=1,kthn=1, return_inds=False):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      kth: 1 is the hardest
      return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """
    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    #dist_ap, relative_p_inds = torch.max(dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    #dist_an, relative_n_inds = torch.min(dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    
    dist_ap = torch.max(dist_mat * is_pos.float().detach(), 1, keepdim=True)[0]
    dist_an = torch.min(torch.max(is_pos.float() * 1000, dist_mat * is_neg.float().detach()), 1, keepdim=True)[0]
    
#ap,inds=torch.topk(dist_mat[is_pos].contiguous().view(N,-1),k=kthp,dim=1,largest=True)

    #an,inds=torch.topk(dist_mat[is_neg].contiguous().view(N,-1),k=kthn,dim=1,largest=False)
   
    #dist_ap=ap[:,kthp-1]
    #dist_an=an[:,kthn-1]
    # shape [N]
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)
    #print(dist_an[0].grad)
    #print(dist_an2[0].grad)
    return dist_ap, dist_an

def distance_mining(dist_mat,labels,cameras):
    assert len(dist_mat.size())==2
    assert dist_mat.size(0)==dist_mat.size(1)
 
    N=dist_mat.size(0)
    
    is_pos=labels.expand(N,N).eq(labels.expand(N,N).t())# & cameras.expand(N,N).eq(cameras.expand(N,N).t())
    is_neg=labels.expand(N,N).ne(labels.expand(N,N).t()) # | cameras.expand(N,N).ne(cameras.expand(N,N).t())
    d1 = torch.max(dist_mat * is_pos.float().detach(), 1, keepdim=True)[0]
    d1=d1.squeeze(1)
    #dist_neg=dist_mat[is_neg].contiguous().view(N,-1)
    d2=d1.new().resize_as_(d1).fill_(0)
    d3=d1.new().resize_as_(d1).fill_(0)
    d2ind=[]
    for i in range(N):
        sorted_tensor,sorted_index=torch.sort(dist_mat[i])
        cam_id=cameras[i]
        B,C=False,False
        for ind in sorted_index:
            if labels[ind]==labels[i]:
                continue
            if B==False and cam_id==cameras[ind]:
                d3[i]=dist_mat[i][ind]
                B=True
            if C==False and cam_id!=cameras[ind]:
                d2[i]=dist_mat[i][ind]
                C=True
                d2ind.append(ind)
            if B and C:
                break
    return d1,d2,d3,d2ind

def hardest_pid_in_other_cameras(dist_mat,labels,cameras):
    N=dist_mat.size(0)
    pid=torch.zeros(N)
    for i in range(N):
        sorted_tensor,sorted_index=torch.sort(dist_mat[i])
        cam_id=cameras[i]
        C=False
        for ind in sorted_index:
            if labels[ind]==labels[i]:
                continue
            if C==False and cam_id!=cameras[ind]:
                C=True
                pid[i]=labels[ind]
            if C:
                break
    return pid.type(torch.LongTensor).cuda()




class DistanceLoss(object):
    """Multi-camera negative loss
        In a mini-batch,
       d1=(A,A'), A' is the hardest true positive. 
       d2=(A,C), C is the hardest negative in another camera. 
       d3=(A,B), B is the hardest negative in same camera as A.
       let d1<d2<d3
    """
    def __init__(self,loader=None,margin=None):
        self.margin=margin
        self.texture_loader=loader
        if margin is not None:
            self.ranking_loss1=nn.MarginRankingLoss(margin=margin[0],reduction="mean")
            self.ranking_loss2=nn.MarginRankingLoss(margin=margin[1],reduction="mean")
        else:
            self.ranking_loss=nn.SoftMarginLoss(reduction="mean")

    def __call__(self,feat,labels,cameras,model=None,paths=None,epoch=0,normalize_feature=False):
        if normalize_feature: # default: don't normalize , distance [0,1]
            feat=normalize(feat,axis=-1)
        dist_mat=euclidean_dist(feat,feat)
        d1,d2,d3,d2ind= distance_mining(dist_mat,labels,cameras)

        y=d1.new().resize_as_(d1).fill_(1)
        if self.margin is not None:
            l1=self.ranking_loss1(d2,d1,y)
            l2=self.ranking_loss2(d3,d2,y)
        else:
            l1=self.ranking_loss(d2-d1,y)
            l2=self.ranking_loss(d3-d2,y)
        loss=l2+l1
        accuracy1=torch.mean((d1<d2).float())
        accuracy2=torch.mean((d2<d3).float()) 
        return loss,accuracy1,accuracy2       
 
class TripletLoss(object):
    """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'."""

    def __init__(self, margin=0.3,kthp=1,kthn=1):
        self.margin = margin
        self.kthp=kthp
        self.kthn=kthn
        self.ranking_loss = nn.MarginRankingLoss(margin=margin, reduction="mean")

    def __call__(self, global_feat, labels, normalize_feature=False):
        #normalize_feature=False # in denfense of the triplet loss
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)
        # dist_mat = cosine_dist(global_feat, global_feat)
        dist_mat = euclidean_dist(global_feat, global_feat)
        dist_ap, dist_an = hard_example_mining(
            dist_mat, labels,kthp=self.kthp,kthn=self.kthn)
        y = dist_an.new().resize_as_(dist_an).fill_(1)
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)

        
        accuracy = torch.mean((dist_ap < dist_an).float())
        return loss, accuracy

