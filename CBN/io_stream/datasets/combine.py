import glob
import re
import numpy as np
import random
from os import path as osp
from collections import defaultdict
from io_stream.data_utils import reorganize_images_by_camera


class Combine(object):
    """
    Market1501
    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: http://www.liangzheng.org/Project/project_reid.html

    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    dataset_dir = 'market'

    def __init__(self, root='data', dataset=None,**kwargs):
        self.name = 'combine'
        self.dataset=dataset
        share_cam = kwargs['share_cam']
        train, num_train_pids, num_train_imgs =[],0,0
        cam_dict=dict()
        num_d = 0
        self.cams_of_dataset=[]
        self.len_of_real_dataset = len(self.dataset[0].train)
        for d in self.dataset:
            d_train = d.train
            cams=set()
            new_train=[]
            for item in d_train:
                img_path, pid , camid = item
                pid+=num_train_pids
                if not (num_d,camid) in cam_dict.keys():
                    cam_dict[(num_d,camid)]=len(cam_dict)
                    cams.add(cam_dict[(num_d,camid)])
                camid= cam_dict[(num_d,camid)] if not share_cam else camid
                new_train.append((img_path,pid,camid))
            self.cams_of_dataset.append(cams)
            num_d +=1    
            train.extend(new_train)
            num_train_pids+=d.num_train_pids
            num_train_imgs+=len(d.train)

        query, num_query_pids, num_query_imgs =[],0,0
        gallery, num_gallery_pids, num_gallery_imgs = [],0,0

        num_total_pids = num_train_pids + num_query_pids
        num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs

        print("=> Combine loaded")
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
        print("length of real data {}".format(self.len_of_real_dataset))
        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids


    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)

        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        if relabel == True:
            self.pid2label = pid2label

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        if relabel:
            self.pid2label = pid2label
        num_pids = len(pid_container)
        num_imgs = len(dataset)
        return dataset, num_pids, num_imgs
