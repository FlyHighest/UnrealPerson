from __future__ import absolute_import

from collections import defaultdict
import random
import numpy as np
import torch
import copy
from torch.utils.data.sampler import Sampler


class IdentitySampler(Sampler):
    def __init__(self, data_source, batch_size, num_instances):
        if batch_size < num_instances:
            raise ValueError('batch_size={} must be no less '
                             'than num_instances={}'.format(batch_size, num_instances))
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances  # approximate
        self.index_dic = defaultdict(list)
        for index, (_, pid, camid) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)
        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []
        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)
        self.length = len(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        return self.length


class IdentityCameraSampler(Sampler):
    def __init__(self, data_source, batch_size, num_instances,cams_of_dataset=None,len_of_real_data=None):
        if batch_size < num_instances:
            raise ValueError('batch_size={} must be no less '
                             'than num_instances={}'.format(batch_size, num_instances))
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances  # approximate
        self.num_cams_per_batch = 8
        self.index_dic = defaultdict(list)
        self.cam_index_dic = dict()
        self.num_pids_per_cam = self.num_pids_per_batch//self.num_cams_per_batch
        for index, (_, pid, camid) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
            if camid not in self.cam_index_dic.keys():
                self.cam_index_dic[camid]=defaultdict(list)
            self.cam_index_dic[camid][pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.cams_of_dataset=cams_of_dataset
        self.len_of_real_data = len_of_real_data

    def __iter__(self):
        final_idxs = []
        length = 2*self.len_of_real_data if self.len_of_real_data is not None else len(self.data_source)
        # F setting
        #length = len(self.data_source)
        while(len(final_idxs) < length):
            if self.cams_of_dataset is not None:
                # C setting
                #c_rnd = np.random.choice(list(self.cam_index_dic.keys()),size=1)[0]
                #for cams_of_data in self.cams_of_dataset:
                #    if c_rnd in cams_of_data:
                #        cams = np.random.choice(list(cams_of_data),size=self.num_cams_per_batch,replace=True)
                #        break
                
                # D setting
                c_rnd = np.random.choice([i for i in range(len(self.cams_of_dataset))],size=1)[0]
                cams = np.random.choice(list(self.cams_of_dataset[c_rnd]),size=self.num_cams_per_batch,replace=True)

                # E setting: data balance, mixed in mini-batches (dontsep)
                #cams0 = np.random.choice(list(self.cams_of_dataset[0]),size=self.num_cams_per_batch//2)
                #cams1 = np.random.choice(list(self.cams_of_dataset[1]),size=self.num_cams_per_batch//2)
                #cams = list(cams0)+list(cams1)

                # F setting databalfix
                # cams = np.random.choice(list(self.cam_index_dic.keys()),size=self.num_cams_per_batch,replace=True)
            else:
                cams = np.random.choice(list(self.cam_index_dic.keys()),size=self.num_cams_per_batch,replace=True)
            for c in cams:
                pids = np.random.choice(list(self.cam_index_dic[c].keys()),size=self.num_pids_per_cam, replace=True)
                for p in pids:
                    idxs =np.random.choice(self.cam_index_dic[c][p],size=self.num_instances,replace=True)
                    random.shuffle(idxs)
                    final_idxs.extend(idxs)
        self.length=len(final_idxs)
        return iter(final_idxs)


    def __iter_old__(self):
        batch_idxs_dict = defaultdict(list)
        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []
        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)
        self.length = len(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        return self.length



class NormalCollateFn:
    def __call__(self, batch):
        img_tensor = [x[0] for x in batch]
        pids = np.array([x[1] for x in batch])
        camids = np.array([x[2] for x in batch])
        return torch.stack(img_tensor, dim=0), torch.from_numpy(pids), torch.from_numpy(np.array(camids))
