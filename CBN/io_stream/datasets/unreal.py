from tqdm import tqdm
import glob
import re
import numpy as np
import random
from os import path as osp
from collections import defaultdict
from io_stream.data_utils import reorganize_images_by_camera
import os

class Unreal(object):
    """
    The first dataset with bad illumanations and low resolutions.
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    

    def __init__(self, root='data',dataset=None, **kwargs):
        self.name = 'unreal_dataset'
        self.dataset_dir = dataset
        if type(self.dataset_dir)==list:
            self.dataset_dir = [osp.join(root,d) for d in self.dataset_dir]
            self.train_dir = [osp.join(d,'images') for d in self.dataset_dir]
        else:
            self.dataset_dir = osp.join(root, self.dataset_dir)
            self.train_dir = osp.join(self.dataset_dir, 'images')
        
        self.num_pids = kwargs['num_pids']
        self.num_cams = kwargs['num_cams']
        self.img_per_person = kwargs['img_per_person']
        self.cams_of_dataset=None
        train, num_train_pids, num_train_imgs = self._process_dir(self.train_dir, relabel=True)
        with open('list_unreal_train.txt','w') as train_file:
            for t in train:
                train_file.write('{} {} {} \n'.format(t[0],t[1],t[2]))
        query, num_query_pids, num_query_imgs = [],0,0 
        gallery, num_gallery_pids, num_gallery_imgs = [],0,0 

        num_total_pids = num_train_pids + num_query_pids
        num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs

        print("=> Unreal {}  loaded".format(dataset))
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # images")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_imgs))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_imgs))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_imgs))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_imgs))
        print("  ------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids

    def size(self,path):
        return os.path.getsize(path)/float(1024)

    def _process_dir(self, dir_path, relabel = True):
        if type(dir_path)!=list:
            dir_path=[dir_path]

        cid_container = set()
        pid_container = set()

        img_paths =[]
        for d in dir_path:
            if not os.path.exists(d):
                assert False, 'Check unreal data dir'

            iii = glob.glob(osp.join(d, '*.*g'))
            print(d,len(iii))
            img_paths.extend( iii )
        
        # regex: scene id, person model version, person id, camera id ,frame id
        pattern = re.compile(r'unreal_v([\d]+).([\d]+)/images/([-\d]+)_c([\d]+)_([\d]+)')
        cid_container = set()
        pid_container = set()
        pid_container_sep = defaultdict(set)
        for img_path in img_paths:
            sid,pv, pid, cid,fid = map(int, pattern.search(img_path).groups())
#            if pv==3 and pid>=1800:
##                continue
#            if pv<3 and pid>=1800: 
#                continue # For training, we use 1600 models. Others may be used for testing later.
            cid_container.add((sid,cid))
            pid_container_sep[pv].add((pv,pid))
        for k in pid_container_sep.keys():
            print("Unreal pids ({}): {}".format(k,len(pid_container_sep[k])))
        print("Unreal cams: {}".format(len(cid_container)))
         # we need a balanced sampler here .
        num_pids_sep = self.num_pids // len(pid_container_sep)
        for k in pid_container_sep.keys():
            pid_container_sep[k]=random.sample(pid_container_sep[k],num_pids_sep) if len(pid_container_sep[k])>=num_pids_sep else pid_container_sep[k]
            for pid in pid_container_sep[k]:
                pid_container.add(pid)

        if self.num_cams!=0:
            cid_container = random.sample(cid_container,self.num_cams)
        print(cid_container)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        cid2label = {cid: label for label, cid in enumerate(cid_container)}
        if relabel == True:
            self.pid2label = pid2label
            self.cid2label = cid2label

        dataset = []
        ss=[]
        for img_path in tqdm(img_paths):
            sid,pv, pid, cid,fid = map(int, pattern.search(img_path).groups())
            if (pv,pid) not in pid_container:continue
            if (sid,cid) not in cid_container:continue
            if relabel: 
                pid = pid2label[(pv,pid)]
                camid = cid2label[(sid,cid)]
#            if self.size(img_path)>2.5:
                dataset.append((img_path, pid, camid))
        print("Sampled pids: {}".format(len(pid_container)))
        print("Sampled imgs: {}".format(len(dataset)))
        if relabel:
            self.pid2label = pid2label
            if len(dataset)>self.img_per_person*self.num_pids:
                dataset=random.sample(dataset,self.img_per_person*self.num_pids)
        
        num_pids = len(pid_container)
        num_imgs = len(dataset)
        print("Sampled imgs: {}".format(len(dataset)))
        return dataset, num_pids, num_imgs


