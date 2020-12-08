import torch, math, random
import torch.nn.functional as F
from torch import nn
import numpy as np

from scipy.spatial.distance import cdist

from .rerank import re_ranking
from .st_distribution import compute_joint_dist, get_st_distribution
from .util import cluster
from .losses_msmt import QueueLoss

class Losses(nn.Module):
    def __init__(self, K, batch_size, bank_size, ann_file, cam_num=8, beta1=0.1, beta2=0.05):
        super(Losses, self).__init__()

        self.loss_src = nn.CrossEntropyLoss()#.cuda()

        self.loss_local= LocalLoss(K=K, beta=beta1)
        self.loss_global = GlobalLoss(K=K, beta=beta2, bank_size=bank_size, cam_num=cam_num, ann_file=ann_file)
        if 'msmt' in ann_file:
            self.loss_global = QueueLoss(K=K,beta=beta2)

    def forward(self, x_src, label_src, x_tar, label_tar, epoch):

        loss_s = self.loss_src(x_src, label_src)

        loss_l = self.loss_local(x_tar,label_tar)
        loss_g = self.loss_global(x_tar, label_tar)

        loss = loss_s + loss_l
        if epoch >= 10:
            loss = loss + loss_g * 0.2

        return loss, loss_s, loss_l, loss_g

    def reset_multi_label(self, epoch):
        if epoch >= 30:
            print('Reset label on target dataset', epoch)
            self.loss_global.reset_label_based_joint_smi()
        elif epoch >=10:
            print('Reset label on target dataset', epoch)
            self.loss_global.reset_label_based_visual_smi()

    def update_memory(self, x_tar, label_tar, epoch):
        self.loss_global.update(x_tar, label_tar, epoch=epoch)
        

class LocalLoss(nn.Module):
    def __init__(self, K=4, batch_size=128, beta=0.1):
        super(LocalLoss, self).__init__()

        self.K = K
        self.beta = beta

    def forward(self, x, idx):
        one_hot_label = []
        label_dic={}
        uni_idx = torch.unique(idx)
        v=[]
        for index in uni_idx:
            if index not in label_dic:
                label_dic[int(index)]=len(label_dic)
            uni_w = x[index==idx]
            uni_w = uni_w.mean(dim=0)
            v.append(uni_w)
       
        for index in idx:
            one_hot_label.append(label_dic[int(index)])

        v=torch.stack(v,dim=0)
        v = F.normalize(v)
        one_hot_label = torch.tensor(one_hot_label).cuda()
        x = F.normalize(x)        
        x = x.mm(v.t())/self.beta
        loss = F.cross_entropy(x, one_hot_label)
        
        return loss

# Global loss
class GlobalLoss(nn.Module):
    def __init__(self, K=4, beta=0.05, cam_num=8, bank_size=114514, ann_file=None):
        super(GlobalLoss, self).__init__()

        self.K = K
        self.beta = beta
        self.cam_num = cam_num
        self.alpha = 0.01
        
        #self.bank = torch.zeros(bank_size, 512).cuda()
        self.bank = torch.rand(bank_size, 512).cuda()
        self.bank.requires_grad = False
        self.labels = torch.arange(0, bank_size).cuda()
        print('Memory bank size', self.bank.size())

        with open(ann_file) as f:
            lines = f.readlines()
            #self.img_list = [os.path.join(dataset_dir, i.split()[0]) for i in lines]
            self.cam_ids = [int(i.split()[2]) for i in lines]
            self.frames = [int(i.split()[3]) for i in lines]

        print('dataset size:', len(self.cam_ids))

    def reset_label_based_visual_smi(self):
        
        bank_fea = self.bank.cpu().data.numpy()
        #fea = fea.numpy()
        dist = cdist(bank_fea, bank_fea)
        dist = np.power(dist,2)
        print('Compute visual similarity')
        rerank_dist = re_ranking(original_dist=dist)

        labels = cluster(rerank_dist)
        num_ids = len(set(labels))

        self.labels = torch.tensor(labels).cuda()
        print('Cluster class num based on visual similarity:', num_ids)

    def reset_label_based_joint_smi(self,):
        
        print('Compute distance based on visual similarity')

        bank_fea = self.bank.cpu().data.numpy()
        dist = cdist(bank_fea, bank_fea)
        dist = np.power(dist,2)

        #Jaccard distance for better cluster result
        dist = re_ranking(original_dist=dist)
        labels = cluster(dist)
        num_ids = len(set(labels))

        print('update st distribution')
        st_distribute = get_st_distribution(self.cam_ids, labels, self.frames, id_num=num_ids, cam_num=self.cam_num)
        print('Compute distance based on joint similarity')
        st_dist = compute_joint_dist(st_distribute, 
            bank_fea,     bank_fea, 
            self.frames,  self.frames, 
            self.cam_ids, self.cam_ids)

        #Jaccard distance for better cluster result
        st_dist = re_ranking(original_dist=st_dist, lambda_value=0.5)
        labels_st = cluster(st_dist)
        num_ids = len(set(labels_st))

        print('Cluster class num based on joint similarity:', num_ids)
        self.labels = torch.tensor(labels_st).cuda()

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
        
        label = w.mm(self.bank.t())
        label = self.multi_class_label(idx)
        targets = []
        for i in range(label.size(0)):
            targets.append(label[i,:])
        targets = torch.stack(targets).detach()
        x = x.mm(self.bank.t())/self.beta
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
        self.bank[uni_idx]= w*(1-momentum) + self.bank[uni_idx]*momentum

        #w = x.view(int(x.size(0)/self.K), self.K, -1)
        #w = w.mean(dim=1)
        #w = F.normalize(w).detach()

        #momentum = min(self.alpha*epoch, 0.8)
        #self.bank[idx] = w*(1-momentum) + self.bank[idx]*momentum
        # for i in range(self.w.size(0)):
        #     self.bank[self.idx[i]] = self.w[i,:]*(1-momentum) + self.bank[self.idx[i]]*momentum
        #     self.bank[self.idx[i]] = F.normalize(self.bank[self.idx[i]], dim=0)
        #     self.bank[idx[i]] /= self.bank[idx[i]].norm()

    def multi_class_label(self, index):
        batch_label = self.labels[index]
        target = (batch_label.unsqueeze(dim=1) == self.labels.t().unsqueeze(dim=0)).float()
        target = F.normalize(target, 1)
        #print target.size()
        return target
