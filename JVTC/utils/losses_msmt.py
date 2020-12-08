import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable, Function
import numpy as np
from . import rerank
import time
from sklearn.cluster import DBSCAN
import random, math
from scipy.spatial.distance import cdist
from .rerank import re_ranking

from .util import cluster
from .st_distribution import compute_joint_dist, get_st_distribution

class BatchLoss(nn.Module):
    def __init__(self, K=4, beta=0.05):
        super(BatchLoss, self).__init__()

        self.K = K
        self.beta = beta

        self.label = []
        for i in range(32):
            for j in range(K):
                self.label.append(i)
        self.label = torch.tensor(self.label).cuda()

    def forward(self, x):

        w = x.view(int(x.size(0)/self.K), self.K, -1)
        w = w.mean(dim=1)
        w = F.normalize(w)

        x = F.normalize(x)        
        x = x.mm(w.t())/self.beta
        loss = F.cross_entropy(x, self.label)
        
        return loss

# Queue loss
class QueueLoss(nn.Module):
    def __init__(self, K=4, beta=0.05, radio=0.2):
        super(QueueLoss, self).__init__()

        self.K = K
        self.beta = beta

        self.len = 32621
        #self.em = torch.zeros(self.len, 512).cuda()
        self.em = torch.rand(self.len, 512).cuda()


        self.em.requires_grad = False
        print(self.em.size())
        self.alpha = 0.01

        self.labels = torch.arange(0,self.len).cuda()
        print('labels', self.labels.size())

        self.cam_ids = []
        self.frames = []
        self.time_slot = []
        for item in open('list_msmt/list_train.txt'):
            item = item.split()[0]
            camera = item.split('_')[2]
            frame = item.split('_')[4]  
            self.cam_ids.append(int(camera))
            self.frames.append(int(frame))
            self.time_slot.append(item.split('_')[3])
            #print frame[0:7]
        self.cam_ids = torch.tensor(self.cam_ids)
        self.frames = torch.tensor(self.frames)

    def reset_label_based_visual_smi(self):
        fea = self.em.cpu().data
        fea = fea.numpy()
        dist = cdist(fea, fea)
        dist = np.power(dist,2)
        rerank_dist = rerank.re_ranking(original_dist=dist)

        rho = 1.6e-3
        tri_mat = np.triu(rerank_dist, 1) # tri_mat.dim=2
        tri_mat = tri_mat[np.nonzero(tri_mat)] # tri_mat.dim=1
        tri_mat = np.sort(tri_mat,axis=None)
        top_num = np.round(rho*tri_mat.size).astype(int)
        eps = tri_mat[:top_num].mean()
        eps = eps*0.8
        

        cluster = DBSCAN(eps=eps,min_samples=1,metric='precomputed', n_jobs=8)
        labels = cluster.fit_predict(rerank_dist)
        num_ids = len(set(labels)) - 1 
        print('eps in cluster: {:.3f}'.format(eps), 'num_ids', num_ids)

        self.labels = torch.tensor(labels).cuda()




    def temporal_dist(self, qf,qc,qfr,gf,gc,gfr,distribution):
        query = qf
        score = np.dot(gf,query)
        alpha=5

        interval = 1.0
        score_st = np.zeros(len(gc))
        for i in range(len(gc)):
            if qfr>gfr[i]:
                diff = qfr-gfr[i]
                hist_ = int(diff/interval)
                pr = distribution[qc-1][gc[i]-1][hist_]
            else:
                diff = gfr[i]-qfr
                hist_ = int(diff/interval)
                pr = distribution[gc[i]-1][qc-1][hist_]
            score_st[i] = pr
        score  = 1-1/(1+np.exp(-alpha*score))*1/(1+2*np.exp(-alpha*score_st))

        return score

    def compute_temporal_dist(self, labels, fea, idx):
        num_ids = len(set(labels))
        frames = self.frames[idx]
        cam_ids = self.cam_ids[idx]

        distribution = get_st_distribution(cam_ids, labels, frames, id_num=num_ids, cam_num=15)
        
        dists = []
        for i in range(len(frames)):
            dist = self.temporal_dist(fea[i], cam_ids[i], frames[i], fea, cam_ids, frames, distribution)
            dist = np.expand_dims(dist, axis=0)
            dists.append(dist)

        dists = np.concatenate(dists, axis=0)
        #print 'num_ids cluster', num_ids, len(idx), 'temporal dist shape', (dists.shape)
        return dists


    def reset_label_based_joint_smi(self):
        temporal_slot = ['0113morning','0113noon','0113afternoon',
                         '0114morning','0114noon','0114afternoon', 
                         '0302morning','0302noon','0302afternoon',
                         '0303morning','0303noon','0303afternoon']

        
        label_sum = 0
        for item in temporal_slot:
            time_divide = []
            for i in range(len(self.time_slot)):
                if self.time_slot[i] == item:
                    time_divide.append(i)

            divide_em = self.em[time_divide]
            fea = divide_em.data
            fea = fea.cpu().numpy()
            
            #dist = fea 
            dist = cdist(fea, fea)
            dist = np.power(dist,2)
            tic = time.time()
            rerank_dist = rerank.re_ranking(original_dist=dist)
            toc = time.time()
            print('    rerank time: ', int(toc-tic),' seconds')

            rho = 1.6e-3*3
            tri_mat = np.triu(rerank_dist, 1) # tri_mat.dim=2
            tri_mat = tri_mat[np.nonzero(tri_mat)] # tri_mat.dim=1
            tri_mat = np.sort(tri_mat,axis=None)
            top_num = np.round(rho*tri_mat.size).astype(int)
            eps = tri_mat[:top_num].mean()#*2
            #eps = eps*2     
            tic= time.time()
            cluster = DBSCAN(eps=eps,min_samples=1,metric='precomputed', n_jobs=8)
            labels = cluster.fit_predict(rerank_dist)  
            num_ids = len(set(labels))
            toc = time.time()

            print('    clustering time: ', int(toc-tic),' seconds')


            st_dist = self.compute_temporal_dist(labels, fea, time_divide)
            #print st_dist.shape
            rerank_st_dist = rerank.re_ranking(original_dist=st_dist)#, lambda_value=0.5)
            tri_mat = np.triu(rerank_st_dist, 1) # tri_mat.dim=2
            tri_mat = tri_mat[np.nonzero(tri_mat)] # tri_mat.dim=1
            tri_mat = np.sort(tri_mat,axis=None)
            top_num = np.round(rho*tri_mat.size).astype(int)
            eps = tri_mat[:top_num].mean()
            #print('eps in temporal cluster: {:.3f}'.format(eps))
            cluster = DBSCAN(eps=eps,min_samples=1,metric='precomputed', n_jobs=8)
            labels = cluster.fit_predict(rerank_st_dist)
            num_ids = len(set(labels))# - 1 
            #print('num_ids temporal cluster', num_ids)


            labels = labels + label_sum
            label_sum = label_sum + num_ids# + 1

            print('eps: %f, num_ids: %d, label sum: %d'%(eps, num_ids, label_sum), item, rerank_st_dist.shape, st_dist.shape)
            #print( label_sum, rerank_dist.shape)

            self.labels[time_divide] = torch.tensor(labels).cuda()

    def forward(self, x, idx):
        uni_idx = torch.unique(idx)
        w = []
        for index in uni_idx:
            uni_w = x[index==idx]
            uni_w = uni_w.mean(dim=0)
            w.append(uni_w)
        w=torch.stack(w)
        w=F.normalize(w)

        x = F.normalize(x)

        label = w.mm(self.em.t())
        label = self.smooth_rerank_label(idx)

        targets = []
        for i in range(label.size(0)):
            targets.append(label[i, :])
        targets = torch.stack(targets).detach()

        x = x.mm(self.em.t())/self.beta
        x = F.log_softmax(x, dim=1)
        loss = - (x * targets).sum(dim=1).mean(dim=0) 

        #self.w = w.detach()
        #self.idx = idx.detach()  
        
        return loss

    def update(self, x, idx, epoch=1):
        uni_idx = torch.unique(idx)
        momentum = min(self.alpha*epoch, 0.8)
        w=[]
        for index in uni_idx:
            uni_w = x[index==idx]
            uni_w = uni_w.mean(dim=0)
            w.append(uni_w)
        w = torch.stack(w)
        w=F.normalize(w).detach()
        self.em[uni_idx]= w*(1-momentum) + self.em[uni_idx]*momentum



    def smooth_rerank_label(self, index):
        batch_label = self.labels[index]
        target = (batch_label.unsqueeze(dim=1) == self.labels.t().unsqueeze(dim=0)).float()
        target = F.normalize(target, 1)
        #print target.size()

        return target

    def smooth_hot(self, inputs, targets, k=6):
        # Sort
        _, index_sorted = torch.sort(inputs, dim=1, descending=True)

        ones_mat = torch.ones(targets.size(0), k).cuda()#to(self.device)
        targets = torch.unsqueeze(targets, 1)
        targets_onehot = torch.zeros(inputs.size()).cuda()#to(self.device)

        weights = F.softmax(ones_mat, dim=1)
        targets_onehot.scatter_(1, index_sorted[:, 0:k], ones_mat * weights)
        targets_onehot.scatter_(1, targets, float(1))

        return targets_onehot


    # def smooth_hot(self, inputs, index, k=6):
    #     _, index_sorted = torch.sort(inputs, dim=1, descending=True)
    #     ones_mat = torch.ones(inputs.size(0), k).cuda()
    #     targets_onehot = torch.zeros(inputs.size()).cuda()

    #     inputs = (inputs > threshold).float()        
    #     targets_onehot.scatter_(1, index_sorted[:, 0:k], ones_mat)
    #     targets_onehot = ((targets_onehot + inputs)>0).float()
    #     #print targets_onehot.sum(dim=1), inputs.size()
    #     targets_onehot = targets_onehot/targets_onehot.sum(dim=1, keepdim=True)

    #     return targets_onehot
